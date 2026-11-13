# -*- coding: utf-8 -*-
"""
fetch_wiki.py — скачивание страниц Minecraft Wiki в data/minecraft/*.md
Запуск: python fetch_wiki.py
"""

import argparse
import re
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
API_EN = "https://minecraft.wiki/api.php"
API_RU = "https://ru.minecraft.wiki/api.php"
OUT_DIR = BASE_DIR / "data" / "minecraft"

# Курируемый список: то, что чаще всего спрашивают новички.
# Порядок важен — эти страницы попадают в базу первыми, до категорий.
PAGES = [
    # Основы, управление, режимы игры
    "Minecraft", "Player", "Controls", "Difficulty", "Game mode",
    "Survival mode", "Creative mode", "Adventure mode", "Hardcore",
    "Health", "Hunger", "Experience", "XP", "Inventory", "Hotbar",
    "Crafting", "Recipe book", "Smelting", "Trading", "Enchanting",
    "Brewing",
    # Первые шаги и туториалы (названия проверены через API: остальных в вики нет)
    "Tutorial:Beginner's guide",
    "Tutorials/Beginner's guide", "Tutorials/Game terms",
    "Tutorials/Hunger management", "Tutorials/Mining",
    "Tutorials/Mining techniques", "Tutorials/Combat",
    "Tutorials/Animal farming", "Tutorials/Crop farming",
    "Tutorials/Automatic farming", "Tutorials/Defeating the Ender Dragon",
    "Tutorials/Nether survival", "Tutorials/End survival",
    "Tutorials/Adventure survival", "Tutorials/Flat survival",
    "Tutorials/Iron golem farming", "Tutorials/Mob farm",
    "Tutorials/Creating a village", "Tutorials/Defeating a village raid",
    "Tutorials/Defeating an End city", "Tutorials/Defeating a Nether fortress",
    "Tutorials/Advancement guide", "Tutorials/Blaze farming",
    "Tutorials/Enderman farming", "Tutorials/Guardian farming",
    "Tutorials/Experience farming", "Tutorials/Curing a zombie villager",
    # Блоки, станции и механизмы
    "Crafting Table", "Furnace", "Blast Furnace", "Smoker", "Campfire",
    "Anvil", "Enchanting Table", "Brewing Stand", "Beacon", "Conduit",
    "Chest", "Ender Chest", "Shulker Box", "Barrel", "Hopper", "Dropper",
    "Dispenser", "Piston", "Sticky Piston", "Observer",
    "Redstone Repeater", "Redstone Comparator", "Rail", "Minecart",
    "Boat", "Ladder", "Fence", "Door", "Trapdoor", "Bed", "Torch",
    "Lantern", "Glass", "Wool", "Mob Spawner", "Monster Spawner",
    # Ресурсы и материалы
    "Diamond", "Diamond Ore", "Deepslate", "Iron Ingot", "Iron Ore",
    "Gold Ingot", "Gold Ore", "Netherite", "Ancient Debris", "Coal",
    "Copper Ingot", "Emerald", "Redstone", "Redstone Dust",
    "Lapis Lazuli", "Quartz", "Amethyst Shard", "Obsidian",
    "Crying Obsidian", "Bedrock", "Stone", "Cobblestone", "Dirt",
    "Grass Block", "Sand", "Gravel", "Wood", "Log", "Planks", "Stick",
    "Water", "Lava", "Bucket", "Water Bucket", "Lava Bucket",
    "Milk Bucket", "Flint and Steel", "Shears", "Fishing Rod",
    "Compass", "Clock", "Map", "Spyglass", "Lead", "Name Tag", "Saddle",
    # Инструменты, оружие, броня
    "Pickaxe", "Axe", "Shovel", "Hoe", "Sword", "Bow", "Crossbow",
    "Trident", "Shield", "Armor", "Helmet", "Chestplate", "Leggings",
    "Boots", "Elytra", "Totem of Undying",
    # Еда
    "Food", "Apple", "Golden Apple", "Bread", "Wheat", "Carrot",
    "Potato", "Baked Potato", "Beetroot", "Melon", "Pumpkin",
    "Cooked Beef", "Raw Beef", "Cooked Porkchop", "Chicken", "Fish",
    "Golden Carrot",
    # Мобы
    "Creeper", "Zombie", "Skeleton", "Spider", "Enderman", "Witch",
    "Slime", "Ghast", "Blaze", "Piglin", "Hoglin", "Zombified Piglin",
    "Strider", "Phantom", "Drowned", "Pillager", "Vindicator", "Evoker",
    "Ravager", "Wither Skeleton", "Magma Cube", "Silverfish",
    "Endermite", "Shulker", "Guardian", "Axolotl", "Bee", "Cat",
    "Ocelot", "Wolf", "Fox", "Horse", "Donkey", "Llama", "Cow", "Pig",
    "Sheep", "Rabbit", "Villager", "Iron Golem", "Snow Golem",
    "Wandering Trader", "Ender Dragon", "Wither", "Elder Guardian",
    "Warden",
    # Мир и измерения
    "Biome", "Overworld", "The Nether", "The End", "Nether Portal",
    "End Portal", "End Portal Frame", "Stronghold", "Village", "Raid",
    "Spawn", "Chunk", "Day", "Night", "Weather", "Thunderstorm",
    # Фермы и механики
    "Farming", "Animal farming", "Crop farming", "Mining", "Combat",
    "Redstone circuit", "Mob farm", "XP farm", "Iron Golem farming",
    "Villager trading hall", "Automatic farming", "Irrigation",
    "Bone Meal", "Composter", "Bee Nest", "Beehive",
    # Зелья и зачарования
    "Potion", "Nether Wart", "Blaze Rod", "Eye of Ender", "Enchantment",
]

# Категории добирают полноту: если в курируемом списке чего-то нет,
# статьи появятся отсюда. Имена проверены через API — общих вроде
# "Building blocks" или "Basic tutorials" в вики нет, они вернули бы пусто.
# Лимит — сколько страниц брать из категории с учётом пагинации.
CATEGORY_LIMITS = {
    "Hostile mobs": 120,
    "Passive mobs": 100,
    "Neutral mobs": 80,
    "Utility blocks": 160,
    "Natural blocks": 140,
    "Light sources": 40,
    "Ore": 40,
    "Food": 80,
    "Tools": 50,
    "Weapons": 40,
    "Armor": 40,
    "Redstone mechanics": 60,
    "Structures": 100,
    "Biomes": 70,
    "Enchantments": 60,
}

# Служебные страницы и другие игры/издания: в базе знаний они только мешают.
EXCLUDE_PREFIXES = (
    "Category:", "Template:", "File:", "Image:", "User:", "Talk:", "Help:",
    "Minecraft Wiki:", "Module:", "MediaWiki:", "Special:", "Portal:",
    "Dungeons:", "Legends:", "Story Mode:", "Earth:", "Movie:", "Novel:",
    "Book:", "Merchandise:", "Education Edition:", "China Edition:",
    "Java Edition:", "Bedrock Edition:", "New Nintendo 3DS Edition:",
    "Realms:", "Launcher:", "Server:", "Development:", "Version/",
    "Java Edition version history", "Bedrock Edition version history",
    "Tutorials/Adding", "Tutorials/Advanced", "Tutorials/Arithmetic",
    "Tutorials/Mod", "Tutorials/Server", "Tutorials/Technical",
    "Tutorials/Command", "Tutorials/Creating a new", "Tutorials/Custom",
    "Tutorials/Data", "Tutorials/Resource", "Tutorials/Sound",
    "Tutorials/Texture", "Tutorials/Updating", "Tutorials/Using",
    "Tutorials/Migrating", "Tutorials/Playing",
)

# Подстроки-фильтры: страницы конкретных изданий и исторические подстраницы
# новичку не нужны, а места в индексе занимают много.
EXCLUDE_SUBSTRINGS = (
    "(Bedrock Edition)", "(Java Edition)", "(Legacy Console Edition)",
    "(Nintendo)", "(Xbox)", "(PlayStation)", "(iOS)", "(Android)",
    "/Before ", "/History", "/Data history", "/Old", " legacy",
    "(Mini-Series)", "(entity)", " UI",
)

# Страницы-указатели, которые попадаются в категориях как «статьи».
EXCLUDE_TITLES = {
    "Allow", "Air", "3D", "Feature", "Structure", "Item", "Mob", "Boss",
    "Tool", "Weapon", "Armor", "Food", "Gameplay", "Biome", "Ore",
    "Redstone components", "Non-renewable resource", "Renewable resource",
}

# Шаблоны рецептов на английской и русской вики.
RECIPE_TEMPLATES = {
    "crafting", "крафт", "smelting", "плавка", "brewing", "варка",
    "stonecutting", "камнерез", "smithing", "кузнечное дело",
    "campfire", "костёр", "костер",
}

# Параметры шаблонов, которые не являются ингредиентами.
RECIPE_SERVICE_KEYS = {
    "type", "тип", "description", "описание", "showname", "name",
    "ignore", "fixed", "category", "nocat", "collapsed", "tool",
    "инструмент", "amount", "head", "tail", "foot", "up", "down",
    "left", "right", "float", "align", "width", "style", "class",
    "debug", "ingredients", "ингредиенты", "recipe", "рецепт",
}

# ================= API =================
session = requests.Session()
session.headers.update({"User-Agent": "NeuroRAG/1.0 (personal project)"})

# Заголовок мусорного раздела — убираем при скачивании (не несут знаний для RAG).
JUNK_HEADINGS = {
    "sounds", "data values", "achievements", "advancements", "gallery",
    "external links", "references", "see also", "navigation", "trivia",
    "unique", "video", "videos", "issues", "history",
}

# Шаблоны-рецепты (в нижнем регистре): если встречаем этот шаблон — парсим ингредиенты.
RECIPE_TEMPLATES = {
    "crafting", "crafting table", "craftingtable",
    "smelting", "blasting", "smoking", "campfire", "blast furnace",
    "brewing", "stonecutting", "smithing",
    "крафт", "плавка", "варка", "камнерез", "кузнечное дело", "обжиг", "жарка",
}

# Параметры шаблонов, которые не являются ингредиентами.
RECIPE_SERVICE_KEYS = {
    "type", "тип", "description", "описание", "showname", "name",
    "ignore", "fixed", "category", "nocat", "collapsed", "tool",
    "инструмент", "amount", "head", "tail", "foot", "up", "down",
    "left", "right", "float", "align", "width", "style", "class",
    "debug", "ingredients", "ингредиенты", "recipe", "рецепт",
    "fuel", "топливо", "name", "output", "выход", "input", "вход",
    "result", "результат", "ingredient", "ингредиент",
}


def polite_delay():
    """Уважаем wiki: пауза между запросами, чтобы не получить 429."""
    time.sleep(0.25)


# ================= МАСШТАБИРУЕМАЯ ПАГИНАЦИЯ =================
def paginated(category: str, limit: int, api_url: str) -> list[str]:
    """
    Загружает все страницы категории (или первые `limit`),
    автоматически обрабатывая cmcontinue для пагинации.
    """
    titles: list[str] = []
    seen: set[str] = set()
    cmcontinue: str | None = None
    total_fetched = 0

    while total_fetched < limit:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmnamespace": "0",
            "format": "json", "cmtype": "page",
        }
        remaining = min(500, limit - total_fetched)
        params["cmlimit"] = str(remaining)
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        r = session.get(api_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        members = data.get("query", {}).get("categorymembers", [])
        if not members:
            break

        for m in members:
            t = m["title"]
            if t in seen or t in EXCLUDE_TITLES:
                continue
            if t.startswith(("Category talk:", "Template:", "File:",
                             "Portal:", "Help:")):
                continue
            # Подстраницы с "/" (кроме Tutorials) — исторические/издание-специфичные
            if "/" in t and not t.startswith("Tutorials/"):
                continue
            # Страницы конкретных изданий — не нужны
            if any(sub in t for sub in EXCLUDE_SUBSTRINGS):
                continue
            if len(t) < 3:
                continue
            seen.add(t)
            titles.append(t)
            total_fetched += 1
            if total_fetched >= limit:
                break

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        polite_delay()

    return titles


# ================= ЯЗЫКОВЫЕ ССЫЛКИ =================
def fetch_langlinks(titles: list[str]) -> dict[str, str]:
    """
    Батч-запрос к EN-вики: возвращает русские названия статей.
    { "English Title": "Русское название" }
    """
    links: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        polite_delay()
        r = session.get(API_EN, params={
            "action": "query", "prop": "langlinks", "lllang": "ru",
            "format": "json", "redirects": 1,
            "titles": "|".join(batch),
        }, timeout=30)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        for p in pages:
            for ll in p.get("langlinks", []):
                if ll.get("lang") == "ru":
                    links[p["title"]] = ll["*"]
                    break
    return links


# ================= ОЧИСТКА ТЕКСТА =================
HEADING_LEVEL_RE = re.compile(r"^(=+)\s*(.*?)\s*=+\s*$")


def _drop_empty_sections(lines: list[str]) -> list[str]:
    """
    Убирает заголовки-листья без текста под ними.
    TextExtracts вырезает таблицы (крафт, история, звуки), и от таких разделов
    остаются пустые оглавления — они засоряют индекс и путают чанкер.
    Родительский заголовок сохраняем: он даёт контекст для дочерних разделов.
    Повторяем проходы: после удаления детей родитель сам может стать листом.
    """
    while True:
        heads = [(i, len(m.group(1)))
                 for i, line in enumerate(lines)
                 if (m := HEADING_LEVEL_RE.match(line.strip()))]
        drop: set[int] = set()

        for pos, (idx, level) in enumerate(heads):
            if pos + 1 < len(heads):
                nxt_idx, nxt_level = heads[pos + 1]
            else:
                nxt_idx, nxt_level = len(lines), 0
            body = "".join(lines[idx + 1:nxt_idx]).strip()
            # Лист: текста нет и вложенных разделов тоже нет
            if not body and nxt_level <= level:
                drop.add(idx)

        if not drop:
            return lines
        lines = [line for i, line in enumerate(lines) if i not in drop]


def clean_extract(text: str) -> str:
    """
    Удаляет мусорные и пустые разделы из извлечённого текста.
    Сохраняет: описание, получение, использование, поведение, спавн, дроп,
    зачарования, зелья, фермы, механики, структуру, мир, характеристики.
    """
    lines = text.split("\n")
    result: list[str] = []
    skip_level = 0

    for line in lines:
        stripped = line.strip()
        m = HEADING_LEVEL_RE.match(stripped)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip().lower()
            # Закрываем мусорный раздел, если ушли на тот же или выше уровень
            if skip_level and level <= skip_level:
                skip_level = 0
            if not skip_level and heading in JUNK_HEADINGS:
                skip_level = level
                continue
        if not skip_level:
            result.append(line)

    result = _drop_empty_sections(result)

    # Сжимаем 3+ пустые строки до двух, чистим невидимые символы
    cleaned: list[str] = []
    blank = 0
    for line in result:
        line = line.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")
        if not line.strip():
            blank += 1
            if blank <= 2:
                cleaned.append(line)
        else:
            blank = 0
            cleaned.append(line)
    return "\n".join(cleaned)


# ================= ИЗВЛЕЧЕНИЕ РЕЦЕПТОВ =================
def _clean_wiki_text(s: str) -> str:
    """Убирает вики-ссылки [[...|...]] → [[...]], шаблоны {{...}}."""
    s = re.sub(r"\[\[([^|\]]+)(?:\|[^]]*)?\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    return s.strip()


def parse_recipes(wikitext: str) -> list[str]:
    """
    Парсит шаблоны-рецепты из wikitext и возвращает строки для RAG.

    Каждая строка — законченное предложение: они склеиваются чанкером в один
    блок, тогда как короткие строки с собственным заголовком отбрасываются
    как огрызки (MIN_CHUNK_CHARS в rag_module).

    Форматы шаблонов на minecraft.wiki:
      Crafting: {{Crafting|A2=Any Planks|B2=Any Planks|A3=Any Planks|B3=Any Planks|Output=Crafting Table}}
      Smelting: {{Smelting|Iron Ore;Deepslate Iron Ore|Iron Ingot|0,7}}
      Крафт (RU): {{Крафт|A2=Любые доски|...|Выход=Верстак}}
    """
    if not wikitext:
        return []

    # Ищем шаблоны {{Name|...}} и {{Name\n|...}} (перенос строки после имени — норма)
    template_re = re.compile(
        r"\{\{\s*([A-Za-z\u0400-\u04FF][A-Za-z\u0400-\u04FF ]*?)(?:\s*\||\s*\n\s*\|)"
        r"((?:[^{}]|\{[^{}]*\})*)\}\}",
        re.DOTALL,
    )
    recipes: list[str] = []

    for m in template_re.finditer(wikitext):
        name = m.group(1).strip().lower()
        if name not in RECIPE_TEMPLATES:
            continue
        params_str = m.group(2)

        # Токенизация параметров: разбиваем по "|", но учитываем вложенность
        params: dict[str, str] = {}
        idx = 0
        for kv in params_str.split("|"):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip().lower()] = _clean_wiki_text(v.strip())
            elif kv:
                params[str(idx)] = _clean_wiki_text(kv.strip())
                idx += 1

        # --- Crafting (A1-C3 сетка) ---
        if name in ("crafting", "крафт", "crafting table", "craftingtable"):
            ingredients: list[str] = []
            for pos in "A1 B1 C1 A2 B2 C2 A3 B3 C3".split():
                v = params.get(pos.lower())
                if v:
                    ingredients.append(v)
            output_val = params.get("output") or params.get("выход")
            if not output_val:
                continue
            output_val = output_val.strip()
            # Убираем дубли, считаем
            counts: dict[str, int] = {}
            for ing in ingredients:
                if ing:
                    counts[ing] = counts.get(ing, 0) + 1
            if not counts:
                continue
            parts = [
                f"{ing} (x{c})" if c > 1 else ing
                for ing, c in sorted(counts.items())
            ]
            recipes.append(f"Crafting recipe for {output_val}: "
                           f"crafted from {', '.join(parts)}.")

        # --- Smelting / Blasting / Smoking / Campfire / Плавка / Обжиг ---
        elif name in (
            "smelting", "blasting", "smoking", "campfire", "blast furnace",
            "плавка", "обжиг", "жарка",
        ):
            # Позиционные: 0=inputs(через ;), 1=output, 2=xp
            # Именованные: input/output/fuel
            input_val = params.get("input") or params.get("вход") or params.get("0")
            output_val = (
                params.get("output") or params.get("выход") or
                params.get("1") or params.get("result") or params.get("результат")
            )
            if not input_val or not output_val:
                continue
            input_val = _clean_wiki_text(input_val)
            output_val = _clean_wiki_text(output_val.strip())
            # inputs разделены ";"
            inputs = [i.strip() for i in input_val.split(";") if i.strip()]
            fuel = params.get("fuel") or params.get("топливо")
            fuel_str = f" Fuel: {fuel}." if fuel else ""
            kind = {
                "smelting": "Smelting", "blasting": "Blasting",
                "smoking": "Smoking", "campfire": "Campfire cooking",
                "blast furnace": "Blasting", "плавка": "Smelting",
                "обжиг": "Smoking", "жарка": "Campfire cooking",
            }.get(name, name.title())
            recipes.append(f"{kind} recipe for {output_val}: "
                           f"smelt {', '.join(inputs)} to get {output_val}."
                           + fuel_str)

        # --- Brewing / Варка (зелья) ---
        elif name in ("brewing", "варка"):
            ingredients: list[str] = []
            for k, v in params.items():
                kl = k.lower()
                if kl in ("result", "результат", "output", "выход",
                          "type", "тип", "level", "уровень",
                          "head", "foot", "name", "showname"):
                    continue
                if v:
                    ingredients.append(v)
            output_val = (
                params.get("result") or params.get("выход") or
                params.get("output")
            )
            if not output_val:
                continue
            output_val = _clean_wiki_text(output_val.strip())
            if not ingredients:
                continue
            parts = [i for i in ingredients if i]
            recipes.append(f"Brewing recipe for {output_val}: "
                           f"brewed from {', '.join(parts)}.")

    return recipes


# ================= ЗАГРУЗКА СТРАНИЦ =================
def fetch_page(title: str) -> str:
    """Загружает одну страницу. Возвращает cleaned extract или пустую строку."""
    polite_delay()
    r = session.get(API_EN, params={
        "action": "query", "prop": "extracts",
        "explaintext": True, "format": "json", "redirects": 1,
        "titles": title,
    }, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page or "invalid" in page:
            return ""
        if "extract" in page:
            return page["extract"]
    return ""


def fetch_page_recipes(title: str, api_url: str = API_EN) -> list[str]:
    """Отдельный запрос parse для извлечения рецептов (EN или RU)."""
    polite_delay()
    r = session.get(api_url, params={
        "action": "parse", "prop": "wikitext",
        "format": "json", "formatversion": "2",
        "page": title,
    }, timeout=30)
    r.raise_for_status()
    wikitext = r.json().get("parse", {}).get("wikitext", "")
    return parse_recipes(wikitext)


# ================= СОХРАНЕНИЕ =================


def safe_name(title: str) -> str:
    """Безопасное имя файла из заголовка страницы."""
    return title.replace("/", "__").replace(":", "_")


def build_existing_set() -> set[str]:
    """Множество safe-имён уже сохранённых файлов для пропуска."""
    existing: set[str] = set()
    for p in OUT_DIR.rglob("*.md"):
        name = p.stem
        if name.endswith("__ru"):
            name = name[:-4]
        existing.add(name)
    return existing


def save_page(title: str, text: str, recipes: list[str],
              lang: str = "en", overwrite: bool = False) -> bool:
    """Сохраняет статью в data/minecraft. Возвращает True при сохранении."""
    safe = safe_name(title)
    ext = ".md" if lang == "en" else "__ru.md"
    path = OUT_DIR / (safe + ext)
    if path.exists() and not overwrite:
        return False

    lines = [f"# {title}\n"]
    if lang == "en":
        lines.append("")
        lines.append(text)
    else:
        lines.append(text)

    if recipes:
        heading = "Recipes" if lang == "en" else "Рецепты"
        lines.append("")
        lines.append(f"== {heading} ==")
        lines.append("")
        lines.append(" ".join(recipes))

    path.write_text("\n".join(lines), encoding="utf-8")
    return True


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
def build_title_set() -> set[str]:
    """Множество уже сохранённых заголовков (EN) для пропуска."""
    existing: set[str] = set()
    for p in OUT_DIR.rglob("*.md"):
        name = p.stem
        if name.endswith("__ru"):
            name = name[:-4]
        existing.add(name.replace("__", "/").replace("_", ":"))
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка данных из Minecraft Wiki (EN + RU) для RAG.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Список категорий (по умолчанию все из CATEGORY_LIMITS)")
    parser.add_argument("--cat-limit", type=int, default=None,
                        help="Лимит страниц на категорию")
    parser.add_argument("--no-recipes", action="store_true",
                        help="Не парсить wikitext для рецептов (быстрее)")
    parser.add_argument("--max-recipes", type=int, default=40,
                        help="Максимум рецептов на страницу (по умолчанию 40)")
    parser.add_argument("--max-pages", type=int, default=500,
                        help="Макс. страниц EN+RU (по умолчанию 500, ~400 RU-переводов)")
    parser.add_argument("--no-ru", action="store_true",
                        help="Не загружать русскую версию wiki")
    parser.add_argument("--force", action="store_true",
                        help="Перезаписывать существующие файлы")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только показать, что будет загружено")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = build_existing_set()

    # --- Курируемый список ---
    curated = [t for t in PAGES if t not in EXCLUDE_TITLES
               and not any(t.startswith(p) for p in EXCLUDE_PREFIXES)]

    # --- Категории ---
    cat_names = args.categories
    if cat_names is None:
        cat_names = [
            "Hostile mobs", "Passive mobs", "Neutral mobs",
            "Utility blocks", "Natural blocks", "Light sources", "Ore",
            "Food", "Tools", "Weapons", "Armor",
            "Redstone mechanics", "Structures", "Biomes", "Enchantments",
        ]
    cat_limits = {
        c: (args.cat_limit if args.cat_limit else CATEGORY_LIMITS.get(c, 80))
        for c in cat_names
    }

    # --- Сбор заголовков ---
    all_titles: list[str] = []
    seen: set[str] = set()

    for t in curated:
        if t not in seen and len(all_titles) < args.max_pages:
            all_titles.append(t)
            seen.add(t)

    for cat, limit in cat_limits.items():
        if len(all_titles) >= args.max_pages:
            break
        print(f"Категория: {cat} (лимит {limit})...")
        titles = paginated(cat, limit, API_EN)
        added = 0
        for t in titles:
            if t not in seen and len(all_titles) < args.max_pages:
                all_titles.append(t)
                seen.add(t)
                added += 1
        print(f"  → добавлено {added} новых страниц")

    print(f"\nВсего заголовков: {len(all_titles)}")

    # --- Dry-run ---
    if args.dry_run:
        print(f"Сухой запуск: {len(all_titles)} страниц")
        for t in all_titles:
            print(f"  - {t}")
        if not args.no_ru:
            print("RU: будет загружена аналогичная выборка через langlinks")
        print("Запустите без --dry-run для загрузки.")
        return

    # --- Проверка существования через API (батчами по 50) ---
    real_titles: list[str] = []
    missing = 0
    for i in range(0, len(all_titles), 50):
        batch = all_titles[i:i + 50]
        polite_delay()
        r = session.get(API_EN, params={
            "action": "query", "format": "json", "redirects": 1,
            "titles": "|".join(batch),
        }, timeout=30)
        pages = r.json().get("query", {}).get("pages", {})
        for p in pages.values():
            if "missing" in p or "invalid" in p:
                missing += 1
            else:
                real_titles.append(p["title"])
    print(f"Проверено: {len(all_titles)}, несущ. {missing}, реальных {len(real_titles)}")

    # --- Загрузка EN ---
    ok, empty, fail = 0, 0, 0
    print("\n--- Загрузка EN ---")
    for title in real_titles:
        sn = safe_name(title)
        if sn in existing and not args.force:
            print(f"[skip] {title}")
            ok += 1
            continue
        try:
            extract = fetch_page(title)
        except Exception as e:
            print(f"[FAIL] {title}: {e}")
            fail += 1
            continue
        if not extract.strip():
            print(f"[empty] {title}")
            empty += 1
            continue
        extract = clean_extract(extract)
        if not extract.strip():
            print(f"[empty after clean] {title}")
            empty += 1
            continue

        recipes: list[str] = []
        if not args.no_recipes:
            try:
                recipes = fetch_page_recipes(title)
                recipes = recipes[:args.max_recipes]
            except Exception:
                pass

        save_page(title, extract, recipes, lang="en", overwrite=args.force)
        size = len(extract) + sum(len(r) for r in recipes)
        print(f"[ok]   {title} ({size} симв.)")
        ok += 1

    print(f"\nИтого EN: скачано {ok}, пустых {empty}, ошибок {fail}")

    # --- Загрузка RU ---
    if not args.no_ru:
        print("\n--- Загрузка RU (языковые ссылки) ---")
        ru_map = fetch_langlinks(real_titles)
        print(f"Найдено русских названий: {len(ru_map)}")

        if not ru_map:
            print("Русская вики не вернула переводов — пропускаем RU.")
        else:
            ru_ok, ru_empty, ru_fail = 0, 0, 0
            for en_title, ru_title in ru_map.items():
                try:
                    polite_delay()
                    r = session.get(API_RU, params={
                        "action": "query", "prop": "extracts",
                        "explaintext": True, "format": "json", "redirects": 1,
                        "titles": ru_title,
                    }, timeout=30)
                    r.raise_for_status()
                    pages = r.json().get("query", {}).get("pages", {})
                    for p in pages.values():
                        if "missing" in p or "invalid" in p:
                            continue
                        if "extract" in p and p["extract"].strip():
                            extract = clean_extract(p["extract"])
                            if not extract.strip():
                                ru_empty += 1
                                continue
                            # рецепты для RU — опционально, может быть дорого
                            recipes: list[str] = []
                            if not args.no_recipes:
                                try:
                                    recipes = fetch_page_recipes(ru_title, API_RU)
                                    recipes = recipes[:args.max_recipes]
                                except Exception:
                                    pass
                            save_page(ru_title, extract, recipes, lang="ru",
                                      overwrite=args.force)
                            print(f"[ru]   {ru_title} ({len(extract)} симв.)")
                            ru_ok += 1
                        else:
                            ru_empty += 1
                except Exception as e:
                    ru_fail += 1
                    if ru_fail <= 5:
                        print(f"[ru FAIL] {en_title} → {ru_title}: {e}")

            print(f"\nИтого RU: скачано {ru_ok}, пустых {ru_empty}, ошибок {ru_fail}")

    # --- Итог ---
    print("\n=== Готово ===")
    file_count = sum(1 for _ in OUT_DIR.rglob("*.md"))
    print(f"Файлов: {file_count}")
    print(f"Запустите: python build_index.py")


if __name__ == "__main__":
    main()
