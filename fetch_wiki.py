# -*- coding: utf-8 -*-
"""
fetch_wiki.py — скачивание страниц Minecraft Wiki в data/minecraft/*.md
Запуск: python fetch_wiki.py
"""

import os
import requests

API = "https://minecraft.wiki/api.php"
OUT_DIR = "./data/minecraft"

# Стартовый набор: можно расширять списком категорий
PAGES = [
    "Creeper", "Zombie", "Skeleton", "Spider", "Enderman", "Ender Dragon",
    "Diamond", "Iron Ingot", "Gold Ingot", "Redstone", "Obsidian",
    "Crafting Table", "Furnace", "Blast Furnace", "Enchanting Table",
    "Anvil", "Brewing Stand", "Potion", "Nether Wart", "Blaze Rod",
    "Eye of Ender", "Ender Chest", "Nether Portal", "The Nether", "The End",
    "Bedrock", "Pickaxe", "Sword", "Bow", "Crossbow", "Shield", "Torch",
    "Water Bucket", "Lava Bucket", "Bucket", "Boat", "Minecart",
    "Villager", "Trading", "Emerald", "XP", "Hunger", "Health",
    "Biome", "Overworld", "Chunk", "Spawn", "Mob Spawner",
    "Tutorial:Beginner's guide",
    "Tutorials/Defeating the Ender Dragon",
]

session = requests.Session()
session.headers.update({"User-Agent": "NeuroRAG/1.0 (personal project)"})


def fetch_page(title: str) -> str:
    r = session.get(API, params={
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "redirects": 1,
        "titles": title,
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "extract" in page:
            return page["extract"]
    return ""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok, empty, fail = 0, 0, 0
    for title in PAGES:
        safe = title.replace("/", "__").replace(":", "_")
        path = os.path.join(OUT_DIR, safe + ".md")
        if os.path.exists(path):
            print(f"[skip] {title}")
            ok += 1
            continue
        try:
            text = fetch_page(title)
        except Exception as e:
            print(f"[FAIL] {title}: {e}")
            fail += 1
            continue
        if not text.strip():
            print(f"[empty] {title}")
            empty += 1
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{text}")
        print(f"[ok]   {title} ({len(text)} симв.)")
        ok += 1
    print(f"\nИтого: скачано {ok}, пустых {empty}, ошибок {fail}")


if __name__ == "__main__":
    main()
