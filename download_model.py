# -*- coding: utf-8 -*-
"""
download_model.py — качает модель эмбеддингов с HuggingFace обычными requests.

Нужно, если штатный загрузчик huggingface_hub (Xet/httpx) не проходит до CDN HF
(обрыв соединения на середине файла). Файлы кладутся в models/<папка>/, и дальше
в настройках RAG_EMBED_MODEL должен быть путь к этой папке, а не имя модели.

Запуск:
    python download_model.py
    python download_model.py --repo BAAI/bge-m3 --out models/bge-m3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_REPO = "BAAI/bge-m3"

session = requests.Session()
session.headers.update({"User-Agent": "NeuroRAG/1.0 (personal project)"})


def repo_files(repo: str) -> list[str]:
    """Список файлов репозитория через публичный API HF."""
    r = session.get(f"https://huggingface.co/api/models/{repo}", timeout=60)
    r.raise_for_status()
    return [s["rfilename"] for s in r.json().get("siblings", [])]


# Что нужно sentence-transformers для загрузки. Всё остальное (onnx, openvino,
# tf, rust-экспорты) — мёртвый вес на сотни мегабайт, его не берём.
NEEDED_NAMES = {
    "config.json", "modules.json", "sentence_bert_config.json",
    "config_sentence_transformers.json", "config_parser_config.json",
    "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.txt", "vocab.json", "merges.txt", "sentencepiece.bpe.model",
    "unigram.json", "added_tokens.json", "chat_template.jinja",
}
NEEDED_PREFIXES = ("1_Pooling/", "2_Dense/", "2_Pooling/")
SKIP_PREFIXES = ("onnx/", "openvino/", "rust/", "flax/", "tf")


def filter_needed(files: list[str]) -> list[str]:
    keep = []
    for f in files:
        if f.startswith(SKIP_PREFIXES):
            continue
        if f.startswith(NEEDED_PREFIXES):
            keep.append(f)
            continue
        name = f.rsplit("/", 1)[-1]
        if name in NEEDED_NAMES or name.endswith((".json", ".txt", ".model")):
            if "/" not in f:            # служебные файлы лежат в корне репо
                keep.append(f)
    # pytorch_model.bin не нужен, если есть safetensors
    if "model.safetensors" in keep:
        keep = [f for f in keep if f != "pytorch_model.bin"]
    return keep


def download_file(repo: str, rel: str, out_dir: Path) -> str:
    dest = out_dir / rel
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {rel}")
        return "skip"

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{rel}"
    with session.get(url, stream=True, timeout=180) as r:
        if r.status_code == 401:
            print(f"[private] {rel} — модель требует HF_TOKEN")
            return "fail"
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                f.write(chunk)
        tmp.replace(dest)

    size = dest.stat().st_size
    print(f"[ok]   {rel} ({size / 1e6:.1f} МБ)")
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Скачать модель эмбеддингов в локальную папку")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Имя репозитория на HF")
    parser.add_argument("--out", default=os.path.join("models", "bge-m3"),
                        help="Куда складывать файлы")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Репозиторий: {args.repo}")
    print(f"Папка:       {out_dir.resolve()}\n")

    try:
        all_files = repo_files(args.repo)
    except Exception as e:
        print(f"Не получил список файлов: {e}")
        sys.exit(1)

    files = filter_needed(all_files)
    skipped = len(all_files) - len(files)
    print(f"Файлов в репозитории: {len(all_files)}, беру {len(files)}, "
          f"пропускаю {skipped} (onnx/tf/openvino-экспорты)\n")

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    for rel in files:
        try:
            counts[download_file(args.repo, rel, out_dir)] += 1
        except Exception as e:
            print(f"[FAIL] {rel}: {e}")
            counts["fail"] += 1

    print(f"\nИтого: скачано {counts['ok']}, уже было {counts['skip']}, "
          f"ошибок {counts['fail']}  ({time.time() - t0:.0f}с)")

    if counts["fail"]:
        print("Что-то не докачалось — повтори команду, докачается с места разрыва.")
        sys.exit(1)

    print(f"\nГотово. В main.py поставь:\n"
          f'    RAG_EMBED_MODEL = r"{out_dir}"\n'
          f"и пересобери индекс: python build_index.py")


if __name__ == "__main__":
    main()
