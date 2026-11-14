# -*- coding: utf-8 -*-
"""
rag_module.py — поиск по локальным текстам об играх (RAG) для голосового ассистента.

Как это устроено:
  data/<игра>/*.md или *.txt   ->  чанки  ->  эмбеддинги  ->  rag_store/<игра>/
  вопрос из STT                ->  эмбеддинг -> косинусное сходство -> top_k чанков
  top_k чанков                 ->  блок текста, который подмешивается в system prompt

Всё локально, без внешних сервисов: индекс — это numpy-матрица нормализованных
векторов плюс JSONL с текстом чанков. Для десятков тысяч чанков этого хватает
с запасом по скорости, FAISS не нужен.
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ================= НАСТРОЙКИ ПО УМОЛЧАНИЮ =================
DEFAULT_DATA_DIR = "data"            # откуда читаем тексты об играх
DEFAULT_STORE_DIR = "rag_store"      # где лежит индекс
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"   # 1024 измерения, RU+EN, лимит 8192 токена

# Целевая длина чанка в символах. Ограничение не косметическое: у моделей
# есть лимит токенов, и всё, что не влезло, молча отбрасывается при эмбеддинге.
# bge-m3 держит 8192 токена (~30k символов), multilingual-MiniLM — только 128
# (~450 символов), поэтому при смене модели проверяйте лимит.
CHUNK_CHARS = 600
CHUNK_OVERLAP_SENTENCES = 1   # сколько последних предложений переносить в следующий чанк

# Разделы без полезного содержимого: оглавления, списки ссылок, техданные.
# Такие чанки похожи на любой вопрос и вытесняют нормальный текст.
# Trivia/Achievements/History сюда НЕ входят — там бывает полезное.
JUNK_HEADINGS = {
    "see also", "gallery", "external links", "references", "navigation",
    "sounds", "data values", "id", "block states", "unique", "video", "notes",
}
MIN_CHUNK_CHARS = 80   # огрызки короче этого бесполезны для поиска

TEXT_EXTENSIONS = {".md", ".txt"}

# Порог косинуса сильно зависит от природы векторов: у семантических моделей
# релевантные чанки обычно 0.45+, у лексического хэша — 0.1-0.35.
MIN_SCORE_BY_BACKEND = {
    "sentence-transformers": 0.45,   # bge-m3: релевантное 0.50+, фон до 0.47
    "llama.cpp": 0.60,               # nomic-embed-text: релевантное 0.73+, фон 0.40-0.50
    "hash-ngram": 0.08,
}


# заголовок раздела: markdown "# Title" или mediawiki "== Title =="
HEADING_RE = re.compile(r"^(?:#{1,6}|={1,6})\s*(.+?)\s*={0,6}$")
# режем по концам предложений, не разрывая слова
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


# ================= ДАННЫЕ =================
@dataclass
class Chunk:
    """Кусок текста, который идёт в индекс."""
    text: str
    source: str
    game: str
    heading: str = ""

    def embed_text(self) -> str:
        """Эмбеддим с заголовком раздела: он сильно повышает попадание по темам."""
        return f"{self.heading}. {self.text}" if self.heading else self.text


@dataclass
class Hit:
    """Найденный чанок со скором."""
    text: str
    source: str
    game: str
    heading: str
    score: float


# ================= ЭМБЕДДЕРЫ =================
class SentenceTransformerEmbedder:
    """Локальные эмбеддинги на GPU через sentence-transformers (рекомендуемый вариант)."""

    backend = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL,
                 device: str = "gpu", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Нужен пакет sentence-transformers: pip install sentence-transformers"
            ) from e

        self.model_name = model_name
        self.batch_size = batch_size

        # Если это путь к локальной папке — грузим строго с диска, без сети.
        # Иначе sentence-transformers может пойти на HF проверять обновления
        # и висеть, если до CDN не пробиться.
        local_dir = Path(model_name)
        is_local = local_dir.is_dir()
        if is_local:
            logger.info(f"Загружаю локальный эмбеддер из {local_dir.resolve()}...")
            self._model = SentenceTransformer(str(local_dir), device=device,
                                              local_files_only=True)
        else:
            logger.info(f"Загружаю эмбеддер {model_name} ({device})...")
            self._model = SentenceTransformer(model_name, device=device)

        # в новых версиях sentence-transformers метод переименован
        dim_fn = getattr(self._model, "get_embedding_dimension", None) \
            or self._model.get_sentence_embedding_dimension
        self.dim = int(dim_fn())
        self.max_seq_length = int(getattr(self._model, "max_seq_length", 0) or 0)
        logger.info(f"Эмбеддер готов, размерность {self.dim}, "
                    f"лимит {self.max_seq_length} токенов.")

    def check_chunk_size(self, chunk_chars: int) -> str | None:
        """
        Сравнивает размер чанка с лимитом модели.
        Возвращает предупреждение или None, если всё влезает.
        """
        if not self.max_seq_length:
            return None
        # грубая оценка: ~3.7 символа на токен для английского текста
        approx_tokens = int(chunk_chars / 3.7)
        if approx_tokens > self.max_seq_length:
            max_chars = int(self.max_seq_length * 3.7)
            return (f"Чанк {chunk_chars} симв. ≈ {approx_tokens} токенов, а модель "
                    f"держит только {self.max_seq_length}. Хвост чанков будет "
                    f"отброшен — уменьшите chunk_chars до ~{max_chars}.")
        return None
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # косинус = обычное скалярное произведение
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    # у этой модели префиксы не нужны — документы и запросы кодируются одинаково
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])


class LlamaCppEmbedder:
    """
    Эмбеддинги через второй llama-server с embedding-моделью (GGUF).

        llama-server -m nomic-embed-text-v1.5.Q5_K_M.gguf --embedding \
                     --pooling mean --port 8081 -c 8192 -b 8192 -ub 8192

    У части моделей (nomic-embed, e5, bge) качество сильно зависит от префиксов:
    документы и запросы кодируются с разными приставками. По умолчанию подставляем
    префиксы nomic — они дают заметный прирост и не мешают другим моделям.
    """

    backend = "llama.cpp"

    def __init__(self, base_url: str = "http://127.0.0.1:8081",
                 model_name: str = "nomic-embed-text-v1.5", batch_size: int = 16,
                 timeout: int = 120,
                 query_prefix: str = "search_query: ",
                 document_prefix: str = "search_document: "):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout
        self.query_prefix = query_prefix or ""
        self.document_prefix = document_prefix or ""
        self.dim = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model_name, "input": batch},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            items = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            out.extend(it["embedding"] for it in items)

        vecs = np.asarray(out, dtype=np.float32)
        self.dim = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.encode([self.document_prefix + t for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([self.query_prefix + text])


class HashingNgramEmbedder:
    """
    Лексический эмбеддер без единой внешней загрузки: подписанный feature hashing
    по словам, основам и символьным n-граммам. Работает на numpy, ничего не качает.

    Это НЕ семантика: он ищет совпадения по форме слов, поэтому подходит, когда
    вопрос и документы на одном языке (русский вопрос по русским статьям). Для
    русских вопросов по английской вики нужен семантический эмбеддер.
    """

    backend = "hash-ngram"

    # частые слова почти не несут смысла и только размывают косинус
    STOPWORDS = frozenset("""
        и в во не ни что как он она оно они мы вы я ты к ко от до из за по на над об
        о а но или же да нет это тот эта эти мой твой свой его её их нас вас мне тебе
        ему для про уже ещё всё так тоже очень если бы ли же ну хорошо быть быть может
        самый такой какой когда где туда сюда тут там там же самый чуть чуть чуть
        the a an and or of to in on for with is are was were be been this that it its
    """.split())

    WORD_RE = re.compile(r"[а-яa-z0-9]+")

    def __init__(self, model_name: str = "hash-ngram-4096", dim: int = 4096,
                 ngram_min: int = 3, ngram_max: int = 4):
        import zlib  # локально: нужен только здесь

        self._crc32 = zlib.crc32
        self.model_name = model_name
        self.dim = int(dim)
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def _features(self, text: str):
        """Генерирует (признак, вес) без IDF — вместо неё чистим стоп-слова."""
        text = text.lower().replace("ё", "е")
        words = [w for w in self.WORD_RE.findall(text) if len(w) > 1]

        feats: dict[str, float] = {}

        def add(key: str, weight: float):
            feats[key] = feats.get(key, 0.0) + weight

        prev = None
        for w in words:
            if w in self.STOPWORDS:
                prev = w
                continue
            add("w:" + w, 1.0)
            if len(w) > 5:                       # грубая основа для русских падежей
                add("s:" + w[:-2], 0.8)
                if len(w) > 7:
                    add("s2:" + w[:-3], 0.6)
            for n in range(self.ngram_min, self.ngram_max + 1):
                if len(w) >= n:
                    for i in range(len(w) - n + 1):
                        add(f"g{n}:" + w[i:i + n], 0.45)
            if prev and prev not in self.STOPWORDS:
                add("b:" + prev + "_" + w, 1.2)  # биграммы ловят устойчивые сочетания
            prev = w

        # сублинейный TF: частотные повторы не должны доминировать
        return [(k, 1.0 + np.log(v)) for k, v in feats.items()]

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        crc = self._crc32
        for i, t in enumerate(texts):
            for key, weight in self._features(t):
                h = crc(key.encode("utf-8")) & 0xFFFFFFFF
                idx = h % self.dim
                sign = 1.0 if (h >> 31) & 1 else -1.0   # signed hashing от коллизий
                out[i, idx] += sign * weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)

    # префиксы тут не нужны: запрос и документ сравниваются по форме слов
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])


def make_embedder(backend: str = "sentence-transformers", **kwargs):
    backend = (backend or "sentence-transformers").lower()
    if backend in ("sentence-transformers", "st", "local"):
        return SentenceTransformerEmbedder(**kwargs)
    if backend in ("llama.cpp", "llamacpp", "server"):
        return LlamaCppEmbedder(**kwargs)
    if backend in ("hash", "hash-ngram", "lexical"):
        kwargs.pop("device", None)
        return HashingNgramEmbedder(**kwargs)
    raise ValueError(f"Неизвестный эмбед-бэкенд: {backend}")


# ================= ЧАНКИНГ =================
def _hard_split(piece: str, limit: int) -> list[str]:
    """Длинные куски режет по предложениям, между соседними — перекрытие."""
    if len(piece) <= limit:
        return [piece]

    sents = [s.strip() for s in SENTENCE_RE.split(piece) if s.strip()]
    if len(sents) == 1:                      # предложений нет вовсе — режем по словам
        words = piece.split()
        step = max(10, limit // 12)
        return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]

    parts, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur)) + len(s) > limit:
            parts.append(" ".join(cur))
            tail = cur[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
            cur = tail + [s]
        else:
            cur.append(s)
    if cur:
        parts.append(" ".join(cur))
    return parts


def split_text(text: str, source: str, game: str,
               chunk_chars: int = CHUNK_CHARS) -> list[Chunk]:
    """
    Разбивает документ на чанки.

    Идём по строкам, запоминаем текущий заголовок раздела, набираем абзацы в чанк
    до chunk_chars. Каждый чанк помнит, из какого он раздела — это и фильтр, и
    подсказка модели.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # 1. блоки: (заголовок раздела, абзац)
    blocks: list[tuple[str, str]] = []
    heading = ""
    paragraph: list[str] = []

    def flush_paragraph():
        if not paragraph:
            return
        body = " ".join(line.strip() for line in paragraph if line.strip()).strip()
        paragraph.clear()
        if body:
            blocks.append((heading, body))

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        m = HEADING_RE.match(stripped)
        if m:
            flush_paragraph()
            heading = m.group(1).strip()
            continue
        paragraph.append(stripped)
    flush_paragraph()

    if not blocks:                      # текст без абзацев и заголовков
        blocks = [("", b.strip()) for b in re.split(r"\n+", text) if b.strip()]

    # 2. атомы: куски не длиннее chunk_chars, мусорные разделы сразу отбрасываем
    atoms: list[tuple[str, str]] = []
    for h, body in blocks:
        if h.strip().lower() in JUNK_HEADINGS:
            continue
        for part in _hard_split(body, chunk_chars):
            atoms.append((h, part.strip()))

    # 3. жадная сборка чанков + перекрытие в 1 предложение
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_head = ""
    cur_len = 0

    for h, part in atoms:
        # смена раздела — не склеиваем, чтобы не смешивать темы
        if cur and (h != cur_head or cur_len + len(part) + 1 > chunk_chars):
            chunks.append(Chunk(text=" ".join(cur), source=source, game=game,
                                heading=cur_head))
            carry: list[str] = []
            # если раздел тот же, тянем последнее предложение в новый чанк для связности
            if h == cur_head and CHUNK_OVERLAP_SENTENCES:
                last = [s.strip() for s in SENTENCE_RE.split(cur[-1]) if s.strip()]
                if last:
                    carry = last[-CHUNK_OVERLAP_SENTENCES:]
            cur = [c for c in carry if len(c) <= chunk_chars]
            cur_len = sum(len(c) + 1 for c in cur)

        cur.append(part)
        cur_len += len(part) + 1
        cur_head = h

    if cur:
        chunks.append(Chunk(text=" ".join(cur), source=source, game=game,
                            heading=cur_head))

    return [c for c in chunks if len(c.text) >= MIN_CHUNK_CHARS]


# ================= ХРАНИЛИЩЕ ИНДЕКСА =================
class RAGIndex:
    """Индекс одной игры: vectors.npy + docs.jsonl + meta.json в rag_store/<game>/."""

    def __init__(self, store_dir: str | Path):
        self.dir = Path(store_dir)
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.docs: list[Chunk] = []
        self.meta: dict = {}
        self.loaded = False

    @property
    def size(self) -> int:
        return len(self.docs)

    def save(self, chunks: list[Chunk], vectors: np.ndarray, embedder) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", vectors.astype(np.float32))

        with open(self.dir / "docs.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(
                    {"text": c.text, "source": c.source, "game": c.game,
                     "heading": c.heading}, ensure_ascii=False) + "\n")

        self.meta = {
            "backend": embedder.backend,
            "embed_model": getattr(embedder, "model_name", ""),
            "dim": int(vectors.shape[1]),
            "count": len(chunks),
            "chunk_chars": CHUNK_CHARS,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self.vectors = vectors.astype(np.float32)
        self.docs = chunks
        self.loaded = True
        logger.info(f"Индекс {self.dir.name}: сохранено {len(chunks)} чанков.")

    def load(self) -> bool:
        vec_path = self.dir / "vectors.npy"
        docs_path = self.dir / "docs.jsonl"
        meta_path = self.dir / "meta.json"
        if not (vec_path.exists() and docs_path.exists()):
            return False

        try:
            self.vectors = np.load(vec_path).astype(np.float32)
            docs = []
            with open(docs_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        docs.append(Chunk(text=d["text"], source=d.get("source", ""),
                                          game=d.get("game", self.dir.name),
                                          heading=d.get("heading", "")))
            self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except Exception:
            logger.exception(f"Не удалось прочитать индекс {self.dir.name} — пересоберите его")
            return False

        if self.vectors.shape[0] != len(docs):
            logger.error(f"Индекс {self.dir.name} повреждён: векторов {self.vectors.shape[0]}, "
                         f"чанков {len(docs)}. Нужна пересборка.")
            return False

        self.docs = docs
        self.loaded = True
        logger.info(f"Индекс {self.dir.name}: загружено {len(docs)} чанков.")
        return True


# ================= МЕНЕДЖЕР =================
class RAGManager:
    """
    Точка входа для main.py:

        rag = RAGManager(top_k=4, min_score=0.35)
        rag.load()                       # при старте
        context = rag.build_context(text)  # перед запросом к LLM
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR,
                 store_dir: str | Path = DEFAULT_STORE_DIR,
                 backend: str = "sentence-transformers",
                 embed_model: str = DEFAULT_EMBED_MODEL,
                 embed_url: str = "http://127.0.0.1:8081",
                 embed_device: str = "cpu",
                 top_k: int = 4,
                 min_score: float | None = None,
                 max_per_source: int = 2,
                 max_context_chars: int = 2400,
                 chunk_chars: int = CHUNK_CHARS):
        self.data_dir = Path(data_dir)
        self.store_dir = Path(store_dir)
        self.backend = backend
        self.embed_model = embed_model
        self.embed_url = embed_url
        self.embed_device = embed_device
        self.top_k = top_k
        self.min_score = min_score
        self.max_per_source = max_per_source
        self.max_context_chars = max_context_chars
        self.chunk_chars = chunk_chars

        self.embedder = None
        self.indexes: dict[str, RAGIndex] = {}
        self._lock = threading.Lock()
        self._warming = False    # идёт фоновая загрузка эмбеддера

    # ---------- подготовка ----------
    def _get_embedder(self):
        """
        Эмбеддер создаётся один раз и переиспользуется (он тяжёлый).
        Если семантическую модель не удалось скачать/загрузить — не падаем:
        переходим на лексический хэш-эмбеддер, база остаётся рабочей.
        """
        if self.embedder is not None:
            return self.embedder

        backend = (self.backend or "sentence-transformers").lower()
        if backend in ("llama.cpp", "llamacpp", "server"):
            self.embedder = make_embedder(backend, base_url=self.embed_url,
                                          model_name=self.embed_model)
        elif backend in ("hash", "hash-ngram", "lexical"):
            self.embedder = make_embedder(backend)
        else:
            # Короткий таймаут на скачивание весов: без него при недоступном HF
            # эмбеддер грузится минут пять вместо тридцати секунд.
            old_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
            try:
                self.embedder = make_embedder(backend, model_name=self.embed_model,
                                              device=self.embed_device)
            except Exception as e:
                logger.warning(f"Семантический эмбеддер недоступен ({type(e).__name__}: {e}).")
                logger.warning("Переключаюсь на лексический hash-ngram: поиск работает без "
                               "интернета, но сопоставляет формы слов, а не смыслы. "
                               "Для русских вопросов по английским статьям нужна "
                               "семантическая модель — скачай её через download_model.py.")
                self.embedder = make_embedder("hash")
            finally:
                if old_timeout is None:
                    os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
                else:
                    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = old_timeout
        return self.embedder

    def warmup(self, background: bool = False):
        """
        Грузит эмбеддер, чтобы первая попытка скачивать модель не повисла
        посреди вопроса.

        background=True — в отдельном потоке: HF при недоступной модели может
        думать несколько минут, и вешать на это старт ассистента нельзя.
        Пока эмбеддер не готов, поиск честно вернёт пустой список.
        """
        if background:
            if self.embedder is not None or self._warming:
                return None
            self._warming = True

            def _job():
                try:
                    backend = self._get_embedder().backend
                    logger.info(f"RAG: эмбеддер готов ({backend!r})")
                except Exception:
                    logger.exception("RAG: эмбеддер не поднялся, база недоступна")
                finally:
                    self._warming = False

            threading.Thread(target=_job, daemon=True).start()
            return None

        return self._get_embedder().backend

    def _effective_min_score(self, explicit: float | None = None) -> float:
        """Порог по умолчанию зависит от бэкенда — у хэша скоры заметно ниже."""
        if explicit is not None:
            return explicit
        if self.min_score is not None:
            return self.min_score
        backend = getattr(self.embedder, "backend", self.backend)
        return MIN_SCORE_BY_BACKEND.get(backend, 0.35)

    def load(self) -> int:
        """Поднимает все готовые индексы из rag_store/. Возвращает число чанков."""
        total = 0
        self.indexes = {}
        if not self.store_dir.exists():
            logger.warning(f"Папки индексов нет: {self.store_dir}. "
                           f"Запустите build_index.py — база будет недоступна.")
            return 0

        for sub in sorted(p for p in self.store_dir.iterdir() if p.is_dir()):
            idx = RAGIndex(sub)
            if idx.load():
                self.indexes[sub.name] = idx
                total += idx.size
        if total == 0:
            logger.warning("Индексы не найдены, RAG выключен до первой сборки.")
        return total

    def ready(self) -> bool:
        return bool(self.indexes) and all(i.size for i in self.indexes.values())

    # ---------- сборка ----------
    def rebuild(self, game: str | None = None) -> dict[str, int]:
        """
        Полная пересборка индексов из data_dir.
        game=None — пересобираем все подпапки, иначе только одну.
        """
        result: dict[str, int] = {}
        embedder = self._get_embedder()

        # Молчаливая обрезка чанков — самая коварная ошибка в RAG: индекс
        # собирается, поиск работает, но модель видит только начало текста.
        check = getattr(embedder, "check_chunk_size", None)
        if check:
            warning = check(self.chunk_chars)
            if warning:
                logger.warning(warning)

        for game_dir, files in self._collect_sources(game):
            chunks: list[Chunk] = []
            for path in files:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    logger.exception(f"Не удалось прочитать {path}")
                    continue
                chunks += split_text(text, path.name, game_dir, self.chunk_chars)

            if not chunks:
                logger.warning(f"В {game_dir} нет подходящего текста")
                result[game_dir] = 0
                continue

            embedder = self._get_embedder()
            vectors = embedder.encode_documents([c.embed_text() for c in chunks])
            RAGIndex(self.store_dir / game_dir).save(chunks, vectors, embedder)
            idx = RAGIndex(self.store_dir / game_dir)
            idx.load()
            self.indexes[game_dir] = idx
            result[game_dir] = len(chunks)
            logger.info(f"Игра {game_dir}: {len(files)} файлов -> {len(chunks)} чанков")

        # игры, где текста не нашлось, из памяти убираем; остальные индексы
        # (например, other games) остаются загруженные
        for g, count in result.items():
            if count == 0:
                self.indexes.pop(g, None)
        return result

    def _collect_sources(self, game: str | None) -> list[tuple[str, list[Path]]]:
        """data/<игра>/файлы; файлы прямо в data/ идут в игру 'default'."""
        if not self.data_dir.exists():
            logger.error(f"Папки с данными нет: {self.data_dir}")
            return []

        groups: list[tuple[str, list[Path]]] = []
        for sub in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            if game and sub.name != game:
                continue
            files = sorted(p for p in sub.rglob("*")
                           if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS)
            if files:
                groups.append((sub.name, files))

        if not game:
            loose = sorted(p for p in self.data_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS)
            if loose:
                groups.append(("default", loose))
        return groups

    # ---------- поиск ----------
    def search(self, query: str, top_k: int | None = None,
               min_score: float | None = None) -> list[Hit]:
        if not query or not query.strip() or not self.ready():
            return []
        if self.embedder is None and self._warming:
            # фоновый прогрев ещё идёт (HF может думать минуты) — не встаём в очередь
            logger.info("RAG: эмбеддер ещё грузится, отвечаю без базы")
            return []

        top_k = self.top_k if top_k is None else top_k

        with self._lock:
            embedder = self._get_embedder()
            min_score = self._effective_min_score(min_score)

            try:
                qvec = embedder.encode_query(query.strip())
            except Exception:
                logger.exception("Не удалось получить эмбеддинг запроса")
                return []
            if qvec.size == 0:
                return []
            qrow = qvec[0]

            hits: list[Hit] = []
            for game, idx in self.indexes.items():
                if idx.vectors.ndim != 2 or idx.vectors.shape[0] == 0:
                    continue
                if idx.vectors.shape[1] != qrow.shape[0]:
                    logger.error(f"Индекс {game} собран другой моделью "
                                 f"(dim={idx.vectors.shape[1]} != {qrow.shape[0]}), "
                                 f"пересоберите его")
                    continue
                sims = idx.vectors @ qrow          # векторы нормализованы -> это косинусы
                k = min(top_k, sims.shape[0])
                for j in np.argpartition(-sims, k - 1)[:k]:
                    score = float(sims[j])
                    if score < min_score:
                        continue
                    c = idx.docs[int(j)]
                    hits.append(Hit(text=c.text, source=c.source, game=c.game,
                                    heading=c.heading, score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        return self._diversify(hits, top_k)

    def _diversify(self, hits: list[Hit], top_k: int) -> list[Hit]:
        """Не более max_per_source чанков с одного файла — иначе контекст однообразный."""
        picked: list[Hit] = []
        per_source: dict[str, int] = {}
        for h in hits:
            key = f"{h.game}/{h.source}"
            if per_source.get(key, 0) >= self.max_per_source:
                continue
            per_source[key] = per_source.get(key, 0) + 1
            picked.append(h)
            if len(picked) >= top_k:
                break
        return picked

    def build_context(self, query: str) -> str:
        """
        Готовит блок материалов для system prompt.
        Пустая строка — ничего релевантного нет, модель отвечает по своим знаниям.
        """
        hits = self.search(query)
        if not hits:
            return ""

        parts = []
        used = 0
        for i, h in enumerate(hits, 1):
            head = f"[{i}] {h.game}: {h.source}"
            if h.heading:
                head += f" — раздел «{h.heading}»"
            block = f"{head}\n{h.text.strip()}"
            if used + len(block) > self.max_context_chars:
                rest = self.max_context_chars - used
                if rest > 200:
                    parts.append(block[:rest].rsplit(" ", 1)[0] + " …")
                break
            parts.append(block)
            used += len(block)
            logger.debug(f"RAG: {h.score:.3f} {h.game}/{h.source}"
                        + (f" «{h.heading}»" if h.heading else ""))

        return "\n\n---\n\n".join(parts)
