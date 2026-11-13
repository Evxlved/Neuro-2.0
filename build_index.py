# -*- coding: utf-8 -*-
"""
build_index.py — сборка / обновление базы знаний RAG.

Команды:
  python build_index.py                           # собрать всё из data/<игра>/
  python build_index.py --list                    # показать состояние
  python build_index.py --game minecraft          # только одну игру

Опции:
  --data-dir        откуда читать тексты   (по умолчанию data)
  --store-dir       куда писать индекс     (по умолчанию rag_store)
  --backend         бэкенд эмбеддингов:
                      sentence-transformers  (по умолчанию, локальная модель на CPU)
                      llama.cpp              (отдельный llama-server)
                      hash                   (лексический поиск без моделей)
  --embed-model     путь к папке модели (по умолчанию models/embeddrus-v0.2)
                    или имя модели на HF
  --embed-device    cpu / cpu:0 / cpu:1 …
  --embed-url       адрес llama-server с embedding-моделью
  --dry-run         только посчитать, не сохранять
  --quiet           минимум логов
"""

import argparse
import logging
import sys
from pathlib import Path

# Убедимся, что корень проекта в sys.path (скрипт может запускаться из любой папки)
sys.path.insert(0, str(Path(__file__).parent))

from rag_module import RAGManager


def main():
    parser = argparse.ArgumentParser(
        description="Сборка базы знаний RAG из текстовых файлов об играх.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data",
                        help="Папка с текстами: data/<игра>/*.md или *.txt")
    parser.add_argument("--store-dir", default="rag_store",
                        help="Папка индексов: rag_store/<игра>/")
    parser.add_argument("--game", default=None,
                        help="Игра: собрать только эту (по умолчанию — все)")
    parser.add_argument("--backend", default="sentence-transformers",
                        choices=["sentence-transformers", "llama.cpp", "hash"],
                        help="Бэкенд эмбеддингов (по умолчанию: sentence-transformers)")
    parser.add_argument("--embed-model",
                        default=str(Path(__file__).parent / "models" / "bge-m3"),
                        help="Путь к локальной папке модели или имя модели на HF")
    parser.add_argument("--embed-device", default="cpu",
                        help="Устройство для sentence-transformers (по умолчанию: cpu)")
    parser.add_argument("--embed-url", default="http://127.0.0.1:8081",
                        help="URL llama-server с embedding-моделью (по умолчанию: "
                             "http://127.0.0.1:8081)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только показать, что было бы собрано")
    parser.add_argument("--quiet", action="store_true",
                        help="Минимум логов в консоль")
    parser.add_argument("--list", action="store_true",
                        help="Показать состояние существующих индексов")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ---------- показать состояние ----------
    if args.list:
        store = Path(args.store_dir)
        if not store.exists():
            print("Индексы не найдены.")
            return
        for sub in sorted(store.iterdir()):
            if not sub.is_dir():
                continue
            meta = sub / "meta.json"
            docs = sub / "docs.jsonl"
            if meta.exists():
                import json
                m = json.loads(meta.read_text(encoding="utf-8"))
                print(f"  {sub.name}: {m['count']} чанков, "
                      f"{m.get('embed_model', '?')}, собран {m['built_at']}")
            else:
                count = docs.read_text(encoding="utf-8", errors="replace").count("\n")
                print(f"  {sub.name}: ~{count} чанков, мета не найдена")
        return

    # ---------- собрать ----------
    manager = RAGManager(
        data_dir=args.data_dir,
        store_dir=args.store_dir,
        backend=args.backend,
        embed_model=args.embed_model,
        embed_url=args.embed_url,
        embed_device=args.embed_device,
    )

    if args.dry_run:
        files = manager._collect_sources(args.game)
        total = 0
        for game_dir, paths in files:
            print(f"  {game_dir}: {len(paths)} файлов")
            total += len(paths)
        print(f"\nИтого файлов: {total}")
        return

    if args.backend == "sentence-transformers" and not Path(args.embed_model).is_dir():
        # покажем предупреждение о первой загрузке модели
        print(f"Модели нет локально ({args.embed_model}) — скачается с HuggingFace.")
        print("Если скачивание не проходит, положи файлы модели в models/embeddrus-v0.2/.")
        print()

    if args.backend == "llama.cpp":
        # без живого сервера сборка упадёт на первом же запросе — скажем сразу
        import requests
        try:
            requests.get(f"{args.embed_url}/health", timeout=5).raise_for_status()
        except Exception as e:
            print(f"Embedding-сервер недоступен по {args.embed_url} ({e}).")
            print("Запустите второй llama-server, например:")
            print(f'  llama-server -m "Neuro LLM\nomic-embed-text-v1.5.Q5_K_M.gguf" '
                  f'--embedding --pooling mean --port 8081 -c 8192 -b 8192 -ub 8192')
            sys.exit(1)

    print(f"Сборка индекса... (backend={args.backend})")
    result = manager.rebuild(args.game)

    if not result:
        print("\nНичего не собрано. Убедитесь, что в data/<игра>/ есть .md или .txt файлы.")
        return

    print()
    for game, count in result.items():
        print(f"  {game}: {count} чанков из файлов")
    total = sum(result.values())
    print(f"\nГотово. Всего чанков: {total}")
    print(f"Индекс: {args.store_dir}/")
    print(f"Запустите main.py для использования RAG.")


if __name__ == "__main__":
    main()
