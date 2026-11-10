import io
import base64
import json
import time
import logging
import queue
import threading
import re
from pathlib import Path

import mss
import requests
import sounddevice as sd
from PIL import Image

from tts_module import TTSManager
from stt_module import STTManager

# ================= НАСТРОЙКИ =================
LLAMA_SERVER = "http://127.0.0.1:8080"

current_dir = Path(__file__).parent
VOSK_MODEL_PATH = str(current_dir / 'models' / 'vosk-model-small-ru-0.22')

MAX_SCREEN_SIDE = 768
JPEG_QUALITY = 70
MAX_TOKENS = 500
STREAM_TIMEOUT = (10, 120)   # (подключение, чтение): зависший стрим умрёт через 2 мин, а не 5
ECHO_PAUSE = 0.5             # пауза после ответа, чтобы эхо не попало в распознаватель

CUT_PUNCT = ".!?…,;:"        # по каким знакам режем поток на фразы
MIN_PHRASE = 30              # минимальная длина фрагмента: короткие огрызки копятся до следующего знака
MAX_TAIL = 150               # хвост совсем без знаков препинания длиннее этого режется по пробелу

GEN_TIMEOUT = 180            # максимум на генерацию ответа
SPOKEN_TIMEOUT = 120         # максимум на ожидание «всё прозвучало»

SYSTEM_PROMPT = (
    "Ты — голосовой ассистент. Пользователь задаёт вопрос, а тебе дано описание "
    "скриншота его экрана. Отвечай кратко, 1–3 предложения, разговорным языком, "
    "только по сути вопроса. Не пересказывай картинку целиком без запроса."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")
# =============================================


# ---------- Скриншот ----------
def take_screenshot() -> str:
    with mss.MSS() as sct:
        shot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    img.thumbnail((MAX_SCREEN_SIDE, MAX_SCREEN_SIDE))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ---------- Разрезание потока на фразы ----------
def extract_phrases(buffer: str,
                    min_phrase: int = MIN_PHRASE,
                    max_tail: int = MAX_TAIL) -> tuple[list[str], str]:
    """
    Достаёт из буфера готовые к озвучке фрагменты, возвращает (фразы, остаток).
    Режет по любому знаку препинания, но только если с прошлого разреза
    накопилось хотя бы min_phrase символов — короткие огрызки склеиваются.
    """
    phrases = []
    start = 0
    for i, ch in enumerate(buffer):
        if ch in CUT_PUNCT:
            piece = buffer[start:i + 1].strip()
            if len(piece) >= min_phrase and re.search(r'\w', piece):
                phrases.append(piece)
                start = i + 1

    buffer = buffer[start:]

    if len(buffer) >= max_tail:
        cut = buffer.rfind(" ")
        if cut > 0:
            phrases.append(buffer[:cut + 1].strip())
            buffer = buffer[cut + 1:]

    return phrases, buffer


def clean_chunk(chunk: str) -> list[str]:
    """
    Чистит чанк от markdown и лишних символов.
    Фрагменты без единой буквы/цифры отбрасываются — Silero на них падает.
    Очень длинные фразы режет по запятым.
    """
    chunk = re.sub(r'[^\w\s.,!?;:—\-()]', '', chunk, flags=re.UNICODE)
    chunk = re.sub(r'\s+', ' ', chunk).strip()
    if not re.search(r'\w', chunk) or len(chunk) < 2:
        return []
    if len(chunk) <= 300:
        return [chunk]
    parts = [p.strip() for p in re.split(r'(?<=[,;:])\s+', chunk)]
    return [p for p in parts if len(p) >= 2 and re.search(r'\w', p)]


# ---------- Потоковый запрос к llama-server ----------
def stream_answer(question: str, image_data_url: str,
                  chunk_queue: queue.Queue, stop_event: threading.Event,
                  stream_finished: threading.Event) -> None:
    """Получает ответ потоком (SSE) и складывает готовые фразы в очередь."""
    payload = {
        "model": "gemma3",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0.6,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    buffer = ""
    try:
        with requests.post(
            f"{LLAMA_SERVER}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=STREAM_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"     # без этого requests декодирует кириллицу как latin-1
            for line in resp.iter_lines(decode_unicode=True):
                if stop_event.is_set():
                    break
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                text_delta = delta.get("content") or ""
                if not text_delta:
                    continue

                buffer += text_delta
                phrases, buffer = extract_phrases(buffer)
                for p in phrases:
                    chunk_queue.put(p)


    except requests.RequestException as e:
        logger.error(f"Ошибка запроса к llama-server: {e}")
        if not buffer:
            buffer = "Не удалось получить ответ от модели."
    except Exception:
        logger.exception("Неожиданная ошибка при получении ответа")
        if not buffer:
            buffer = "Внутренняя ошибка ассистента."
    finally:
        tail = buffer.strip()
        if tail and re.search(r'\w', tail):
            chunk_queue.put(tail)
        chunk_queue.put(None)          # маркер конца — всегда, при любом исходе
        stream_finished.set()          # событие после маркера, не раньше


# ---------- Поток синтеза: чанки -> аудио (по порядку) ----------
def tts_worker(tts: TTSManager, chunk_queue: queue.Queue,
               stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        chunk = chunk_queue.get()
        try:
            if chunk is None:
                break

            for fragment in clean_chunk(chunk):
                if stop_event.is_set():
                    break
                try:
                    with tts.model_lock:
                        audio = tts.model.apply_tts(
                            text=fragment,
                            speaker=tts.speaker,
                            sample_rate=tts.sample_rate,
                            put_accent=True,
                            put_yo=True,
                        )
                    if stop_event.is_set():
                        break
                    tts.audio_queue.put(audio, block=True)
                except Exception:
                    logger.exception(f"Ошибка синтеза чанка: {fragment[:60]!r}")
        finally:
            chunk_queue.task_done()

    # если вышли по stop_event — добиваем очередь
    while True:
        try:
            chunk_queue.get_nowait()
            chunk_queue.task_done()
        except queue.Empty:
            break


def wait_until_spoken(tts: TTSManager, stt: STTManager, chunk_queue: queue.Queue,
                      stream_finished: threading.Event) -> None:
    """Ждёт конца генерации и воспроизведения. Не висит вечно: на каждом этапе таймаут."""
    while stt.get_text() is not None:
        pass

    if not stream_finished.wait(timeout=GEN_TIMEOUT):
        logger.error("Генерация не завершилась вовремя")
        return

    deadline = time.time() + SPOKEN_TIMEOUT
    while time.time() < deadline:
        while stt.get_text() is not None:
            pass   # выкидываем эхо собственного голоса

        if (chunk_queue.unfinished_tasks == 0
                and tts.audio_queue.empty()
                and not getattr(tts, "playing", False)):
            break
        time.sleep(0.1)
    else:
        logger.warning("Таймаут ожидания озвучки, возвращаюсь к слушанию")


def main():
    tts = TTSManager(language="ru", model_id="v5_5_ru", device="cpu",
                     speaker="baya", sample_rate=48000, volume=0.5)

    stt = STTManager(model_path=VOSK_MODEL_PATH, sample_rate=16000)
    stt.start_listening()

    tts.speak("Ассистент готов к работе.")
    logger.info("Готов. Задайте вопрос голосом.")

    try:
        while True:
            text = stt.get_text()
            if text is None:
                time.sleep(0.1)
                continue

            logger.info(f"Распознано: {text}")

            if any(w in text for w in ("выход", "завершить работу", "отключись")):
                tts.speak("До свидания.")
                tts.wait_idle(timeout=5)
                break

            tts.wait_idle()               # дожидаемся синтеза и воспроизведения

            logger.info("Делаю скриншот...")
            image = take_screenshot()

            chunk_queue = queue.Queue()
            stop_event = threading.Event()
            stream_finished = threading.Event()

            worker = threading.Thread(
                target=tts_worker, args=(tts, chunk_queue, stop_event), daemon=True)
            worker.start()

            gen_thread = threading.Thread(
                target=stream_answer,
                args=(text, image, chunk_queue, stop_event, stream_finished),
                daemon=True)
            gen_thread.start()

            wait_until_spoken(tts, stt, chunk_queue, stream_finished)
            logger.info("Ответ озвучен полностью.")

            time.sleep(ECHO_PAUSE)        # даём эху затихнуть до возврата к слушанию
            stt.reset()                   # сброс распознавателя после эха собственного ответа
            worker.join(timeout=2)

    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    finally:
        stt.stop_listening()
        tts.stop()


if __name__ == "__main__":
    main()
