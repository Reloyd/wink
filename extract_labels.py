# extract_labels.py — полный файл с улучшенным парсером заголовков
import re
import csv
import sys
from docx import Document
from normalize import normalize_headings, normalize_scene_heading_strict

# ===== УЛУЧШЕННЫЙ REGEX ДЛЯ РАЗБИВКИ НА СЦЕНЫ =====
COMPREHENSIVE_SPLIT = re.compile(
    r'(?=^\s*'
    r'(?:'
    # БЛОК 1: Стандартный формат (с дефисом)
    r'(?:\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
    r'(?:\d{1,2}-[ЕE]\.?)?\s*'
    r'(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?|И/Н|I/E)\s+'
    r'[^\n]{2,140}?'
    r'\s*[-–—]\s*'
    r'(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|EVENING|MORNING|РЕЖИМ|РАССВЕТ)(?:\s+\d+)?'
    r'|'
    # БЛОК 2: БЕЗ дефиса (точка или пробел перед временем)
    r'(?:\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
    r'(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'  # было \s+, стало \s*
    r'[^\n]{2,200}?'
    r'\.?\s+'
    r'(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)(?:\s+\d+)?'
    r'\.?\s*$'
    r'|'
    # БЛОК 3: Слитное написание "ЛЕС.ОПУШКА НОЧЬ"
    r'(?:\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
    r'(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'  # было \s+, стало \s*
    r'[A-ZА-ЯЁ]+\.[A-ZА-ЯЁ][^\n]{1,100}?'
    r'\s+(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|РЕЖИМ|РАССВЕТ)(?:\s+\d+)?'
    r'|'
    # БЛОК 4: Номер.тип ЛОКАЦИЯ - ВРЕМЯ
    r'\d+\.\s*(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s+[^\n]{2,120}\s*[-–—]\s*'
    r'(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|РЕЖИМ|РАССВЕТ)(?:\s+\d+)?'
    r'|'
    # БЛОК 5: Служебные маркеры
    r'(?:СЦЕНА|СЕРИЯ|ТИТРЫ)\s*\d*'
    r'|'
    # БЛОК 6: Формат "номер. тип. ЛОКАЦИЯ. ВРЕМЯ" (с точками вместо дефисов)
    r'\d+\.\s*(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'  # было \s+, стало \s*
    r'[^\n]{2,200}?\.\s*'
    r'(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)\b'
    r'|'
    # БЛОК 7: Формат "номер. тип. ЛОКАЦИЯ. ПОДЛОКАЦИЯ" (без времени, для сцен 12, 13)
    r'(?:\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
    r'(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'
    r'[A-ZА-ЯЁ][^\n.]{2,100}?\.[A-ZА-ЯЁ][^\n]{2,100}?'
    r'(?:\s*/\s*[A-ZА-ЯЁ][^\n]{2,100}?)?'
    r'|'
    # БЛОК 8: Слитное без пробелов "номер. тип.ЛОКАЦИЯ" или с / (для 10, 15)
    r'(?:\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
    r'(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'
    r'[A-ZА-ЯЁ][^\n\s]{4,120}?(?:\s*/\s*[A-ZА-ЯЁ][^\n]{2,100}?)?'
    r'(?=\s*(?:ДЕНЬ|НОЧЬ|\n))'
    r'|'
    # БЛОК 9: "номер. НАТ.У ЦИРКА. ДЕНЬ" (точки вместо дефисов, для 3, 4, 7)
    r'\d+\.\s*(?:ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'
    r'[A-ZА-ЯЁ][^\n.]{2,80}?\.[A-ZА-ЯЁ][^\n]{2,80}?'
    r'(?:\s*\.\s*(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT)\b)?'
    r')'
    r')',
    re.IGNORECASE | re.MULTILINE
)

def split_scenes(text: str):
    """Разбивка с многопаттернным regex + фильтр коротких фрагментов"""
    parts = re.split(COMPREHENSIVE_SPLIT, text)
    scenes = []
    for p in parts:
        p = p.strip()
        word_count = len(p.split())
        # Минимум 5 слов ИЛИ наличие характерных действий
        has_action = bool(re.search(
            r'\b(входит|выходит|говорит|смотрит|берёт|идёт|садится|стоит|открывает|закрывает)\b',
            p, re.IGNORECASE
        ))
        if word_count >= 5 or (word_count >= 3 and has_action):
            scenes.append(p)
    return scenes

# ===== МНОЖЕСТВЕННЫЕ ПАТТЕРНЫ ДЛЯ ПАРСИНГА ЗАГОЛОВКОВ =====
HEADER_PATTERNS = [
    # Паттерн 1: Стандартный с дефисом "1-2. ИНТ. ЛОКАЦИЯ - НОЧЬ"
    re.compile(
        r'^\s*(?P<scene_no>\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
        r'(?P<period>\d{1,2}-[ЕE]\.?)?\s*'
        r'(?P<place_type>ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s+'
        r'(?P<location>[^-–—:\n]{2,140}?)\s*[-–—]\s*'
        r'(?P<tod>ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)(?:\s+\d+)?',
        re.IGNORECASE
    ),
    
    # Паттерн 2: БЕЗ дефиса "1-2. ИНТ. ЛОКАЦИЯ НОЧЬ"
    re.compile(
        r'^\s*(?P<scene_no>\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
        r'(?P<period>\d{1,2}-[ЕE]\.?)?\s*'
        r'(?P<place_type>ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s+'
        r'(?P<location>(?:[A-ZА-ЯЁ][^\n]{0,100}?\.)?[A-ZА-ЯЁ][^\n]{1,100}?)'
        r'\s+(?P<tod>ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)(?:\s+\d+)?\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    ),
    
    # Паттерн 3: Слитное "ЛЕС.ОПУШКА НОЧЬ"
    re.compile(
        r'^\s*(?P<scene_no>\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?\.?\s*'
        r'(?P<place_type>ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s+'
        r'(?P<location>[A-ZА-ЯЁ][^\s]{2,40}\.[A-ZА-ЯЁ][^\s]{2,80}|[A-ZА-ЯЁ][^\n]{2,100}?)'
        r'\s+(?P<tod>ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|РЕЖИМ|РАССВЕТ)(?:\s+\d+)?',
        re.IGNORECASE
    ),
    
    # Паттерн 4: "номер.тип ЛОКАЦИЯ - ВРЕМЯ" (старый формат)
    re.compile(
        r'^\s*(?P<scene_no>\d+)\.\s*'
        r'(?P<place_type>ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)?\s*'
        r'(?P<location>[^-–—\n]{2,120}?)\s*[-–—]\s*'
        r'(?P<tod>ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ)\b',
        re.IGNORECASE
    ),
    
    # Паттерн 5: "номер. тип. ЛОКАЦИЯ. ВРЕМЯ" (с точками) — КЛЮЧЕВОЙ!
    re.compile(
        r'^\s*(?P<scene_no>\d+)\.\s*'
        r'(?P<place_type>ИНТ\.|НАТ\.|INT\.|EXT\.)\s*'   # \s* вместо \s+
        r'(?P<location>[^\n]{2,200}?)\.\s*'
        r'(?P<tod>ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)\b',
        re.IGNORECASE
    ),
    
    re.compile(
        r'^\s*(?P<scene_no>\d+)\.\s*'
        r'(?P<place_type>ИНТ\.|НАТ\.|INT\.|EXT\.)\s*'
        r'(?P<location>[^\n]{2,150}?)\.\s*'
        r'(?P<tod>ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)\b',
        re.IGNORECASE
    ),

    # Паттерн 6: Только ЛОКАЦИЯ - ВРЕМЯ (без номера/типа)
    re.compile(
        r'^\s*(?P<location>[A-ZА-ЯЁ][^\n-–—]{2,100}?)\s*[-–—]\s*'
        r'(?P<tod>ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ)\b',
        re.IGNORECASE
    ),

    # Паттерн 7: "номер. ИНТ. ЛОКАЦИЯ. ПОДЛОКАЦИЯ" (без времени, сцены 12, 13)
    re.compile(
        r'^\s*(?P<scene_no>\d+(?:\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?)?)?\.?\s*'
        r'(?P<place_type>ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'
        r'(?P<location>[A-ZА-ЯЁ][^\n.]{2,100}?\.[A-ZА-ЯЁ][^\n]{2,100}?)'
        r'(?:\s*/\s*[A-ZА-ЯЁ][^\n]{2,100}?)?',
        re.IGNORECASE | re.MULTILINE
    ),

    # Паттерн 8: "номер. НАТ.У ЦИРКА. ДЕНЬ" (точки + время, сцены 3, 4)
    re.compile(
        r'^\s*(?P<scene_no>\d+)\.\s*'
        r'(?P<place_type>ИНТ\.?|НАТ\.?|INT\.?|EXT\.?)\s*'
        r'(?P<location>[A-ZА-ЯЁ][^\n.]{2,60}?\.)'
        r'[A-ZА-ЯЁ][^\n]{2,80}?'
        r'(?:\s*\.\s*(?P<tod>ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|DAY|NIGHT|РЕЖИМ|РАССВЕТ)\b)?',
        re.IGNORECASE
    ),
]

def normalize_place_type(raw: str) -> str:
    """Нормализация типа места с учётом опечаток"""
    if not raw:
        return ""
    low = raw.lower().replace(".", "").strip()
    mapping = {
        "инт": "ИНТ.", "int": "INT.", "і": "ИНТ.", "ін": "ИНТ.",
        "нат": "НАТ.", "ext": "EXT.", "nat": "НАТ.",
        "и/н": "И/Н", "i/e": "I/E"
    }
    return mapping.get(low, raw.upper())

def normalize_tod(raw: str) -> str:
    """Нормализация времени суток"""
    if not raw:
        return ""
    low = raw.lower().strip()
    mapping = {
        "день": "ДЕНЬ", "day": "ДЕНЬ",
        "ночь": "НОЧЬ", "night": "НОЧЬ",
        "вечер": "ВЕЧЕР", "evening": "ВЕЧЕР",
        "утро": "УТРО", "morning": "УТРО",
        "режим": "РЕЖИМ",  # ← ДОБАВЛЕНО
        "рассвет": "РАССВЕТ",
        "закат": "ЗАКАТ",
        "сумерки": "СУМЕРКИ"
    }
    return mapping.get(low, raw.upper())

def heuristic_parse(line: str):
    """Эвристический парсер для нестандартных заголовков"""
    result = {"scene_no": "", "period": "", "place_type": "", "location": "", "tod": ""}
    
    # Номер сцены
    num_match = re.search(r'\b(\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?|\d+)\b', line)
    if num_match:
        result["scene_no"] = num_match.group(1).strip()
    
    # Тип места
    type_match = re.search(r'\b(ИНТ\.?|НАТ\.?|INT\.?|EXT\.?|[иінат]+\.?)\b', line, re.IGNORECASE)
    if type_match:
        result["place_type"] = normalize_place_type(type_match.group(1))
    
    # Время суток (РАСШИРЕННЫЙ СПИСОК)
    tod_match = re.search(
        r'\b(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|день|ночь|вечер|утро|DAY|NIGHT|РЕЖИМ|режим|РАССВЕТ|рассвет|ЗАКАТ|СУМЕРКИ)(?:\s+\d+)?',
        line,
        re.IGNORECASE
    )
    if tod_match:
        result["tod"] = normalize_tod(tod_match.group(1).split()[0])
        # Локация — текст между типом и временем
        if result["place_type"] and result["tod"]:
            loc_pattern = rf'{re.escape(result["place_type"])}\s*(.+?)\s*[-–—:]\s*{re.escape(result["tod"])}'
            loc_match = re.search(loc_pattern, line, re.IGNORECASE)
            if loc_match:
                result["location"] = loc_match.group(1).strip().strip('.')
        elif result["tod"]:
            # Только время есть — берём всё до разделителя
            loc_match = re.search(r'(.+?)\s*[-–—:]\s*' + re.escape(result["tod"]), line, re.IGNORECASE)
            if loc_match:
                result["location"] = loc_match.group(1).strip().strip('.')
    
    return result

def parse_header(scene_text: str):
    """Fuzzy-парсинг с fallback через несколько паттернов"""
    lines = scene_text.splitlines()
    first_line = lines[0] if lines else scene_text[:200]
    
    # Попытка по порядку паттернов
    for pattern in HEADER_PATTERNS:
        m = pattern.search(first_line)
        if m:
            return {
                "scene_no": (m.groupdict().get("scene_no") or "").strip(),
                "period": (m.groupdict().get("period") or "").strip(),
                "place_type": normalize_place_type(m.groupdict().get("place_type") or ""),
                "location": (m.groupdict().get("location") or "").strip().strip('. '),
                "tod": normalize_tod(m.groupdict().get("tod") or "")
            }
    
    # Fallback: эвристический парсинг
    return heuristic_parse(first_line)

# ===== РАЗМЕТКА (LABELS) =====
LABEL_RE = re.compile(r'\[\s*(?:Labels|МЕТКИ)\s*:\s*([^\]]+)\]', re.IGNORECASE)

MAP_KEY = {
    "v": "violence", "p": "profanity", "s": "sexual", "a": "alcohol_drugs", "sc": "scary",
    "насилие": "violence", "брань": "profanity", "секс": "sexual",
    "алкоголь": "alcohol_drugs", "страшное": "scary"
}

NORM_SEV = {
    "none": "None", "mild": "Mild", "moderate": "Moderate", "severe": "Severe",
    "нет": "None", "лёгкое": "Mild", "легкое": "Mild",
    "среднее": "Moderate", "жёсткое": "Severe", "жесткое": "Severe"
}

def normalize_text(text: str) -> str:
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = re.sub(r"[ \t]*\\\\\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("{.smallcaps}", "")
    return text

def read_text(path: str) -> str:
    if path.lower().endswith(".docx"):
        doc = Document(path)
        raw = "\n".join(p.text or "" for p in doc.paragraphs)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    raw = normalize_text(raw)
    raw = normalize_headings(raw)
    return raw

def parse_label_line(line: str):
    labels = {k: "None" for k in ["violence", "sexual", "profanity", "alcohol_drugs", "scary"]}
    pairs = [p.strip() for p in line.split(",") if p.strip()]
    for p in pairs:
        if "=" not in p:
            continue
        k, v = [x.strip() for x in p.split("=", 1)]
        k = MAP_KEY.get(k.lower(), k.lower())
        v = NORM_SEV.get(v.lower(), v)
        if k in labels:
            labels[k] = v
    return labels

def extract_labels_from_scene(scene_text: str):
    labels = {k: "None" for k in ["violence", "sexual", "profanity", "alcohol_drugs", "scary"]}
    m = LABEL_RE.search(scene_text)
    if m:
        labels = parse_label_line(m.group(1))
    return labels, []

def main(input_path: str, out_csv: str = "labels.csv"):
    """
    Извлечение меток из сценария с улучшенным парсингом заголовков.
    """
    # Читаем и нормализуем весь текст
    text = read_text(input_path)
    text = normalize_headings(text)
    
    # Разбиваем на сцены
    scenes = split_scenes(text)
    
    # Открываем CSV для записи
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        
        # Записываем заголовки столбцов
        w.writerow([
            "scene_no", "place_type", "location", "tod",
            "has_violence", "sev_violence",
            "has_sexual", "sev_sexual",
            "has_profanity", "sev_profanity",
            "has_alcohol_drugs", "sev_alcohol_drugs",
            "has_scary", "sev_scary"
        ])
        
        wrote = 0
        
        # Открываем лог для проблемных заголовков
        with open("bad_headers.log", "w", encoding="utf-8") as blog:
            for i, s in enumerate(scenes, 1):
                # ===== КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Нормализуем заголовок отдельно =====
                first_line = s.splitlines()[0] if s.splitlines() else s[:200]
                normalized_header = normalize_scene_heading_strict(first_line)
                
                # Парсим нормализованный заголовок
                try:
                    meta = parse_header(normalized_header)
                except Exception as e:
                    # Логируем ошибку парсинга
                    blog.write(f"[PARSE ERROR] {first_line}\n")
                    blog.write(f"  Error: {str(e)}\n")
                    meta = {"scene_no": "", "place_type": "", "location": "", "tod": ""}
                
                # Извлекаем метки из текста сцены
                labels, _ = extract_labels_from_scene(s)
                
                # Вычисляем флаги наличия контента
                has = {
                    "violence": int(labels["violence"] != "None"),
                    "sexual": int(labels["sexual"] != "None"),
                    "profanity": int(labels["profanity"] != "None"),
                    "alcohol_drugs": int(labels["alcohol_drugs"] != "None"),
                    "scary": int(labels["scary"] != "None"),
                }
                
                # Проверяем, пустая ли строка (нет метаданных И нет меток)
                meta_empty = not (
                    meta.get("scene_no") or 
                    meta.get("place_type") or 
                    meta.get("location") or 
                    meta.get("tod")
                )
                all_none = sum(has.values()) == 0
                
                # Пропускаем сцены без данных
                if meta_empty and all_none:
                    blog.write(f"[SKIPPED - EMPTY] {first_line[:120]}\n\n")
                    continue
                
                # Логируем успешно распарсенные заголовки с пустой локацией
                if meta.get("place_type") and not meta.get("location"):
                    blog.write(f"[WARNING - NO LOCATION] Original: {first_line}\n")
                    blog.write(f"  Normalized: {normalized_header}\n")
                    blog.write(f"  Parsed: {meta}\n\n")
                
                # Записываем строку в CSV
                w.writerow([
                    meta.get("scene_no", ""),
                    meta.get("place_type", ""),
                    meta.get("location", ""),
                    meta.get("tod", ""),
                    has["violence"], labels["violence"],
                    has["sexual"], labels["sexual"],
                    has["profanity"], labels["profanity"],
                    has["alcohol_drugs"], labels["alcohol_drugs"],
                    has["scary"], labels["scary"],
                ])
                wrote += 1
    
    # Итоговая статистика
    print(f"✅ OK: нарезано сцен: {len(scenes)}; записано в CSV: {wrote}")
    print(f"📊 Пропущено (пустые): {len(scenes) - wrote}")
    print(f"📋 Проверь bad_headers.log для диагностики")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "annotated_script.docx"
    main(inp)
