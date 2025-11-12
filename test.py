# test.py — инференс с улучшенным парсером заголовков
import os
import re
import json
import pickle
import numpy as np
import torch
import pdfplumber
from docx import Document
from transformers import AutoTokenizer, AutoModel
from normalize import normalize_headings
from embeddings import scene_vector

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
    """Разбивка с многопаттернным regex"""
    parts = re.split(COMPREHENSIVE_SPLIT, text)
    scenes = []
    for p in parts:
        p = p.strip()
        word_count = len(p.split())
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
    if not raw:
        return ""
    low = raw.lower().strip()
    mapping = {
        "день": "ДЕНЬ", "day": "ДЕНЬ",
        "ночь": "НОЧЬ", "night": "НОЧЬ",
        "вечер": "ВЕЧЕР", "evening": "ВЕЧЕР",
        "утро": "УТРО", "morning": "УТРО"
    }
    return mapping.get(low, raw.upper())

def heuristic_parse(line: str):
    result = {"scene_no": "", "period": "", "place_type": "", "location": "", "tod": ""}
    
    num_match = re.search(r'\b(\d+\s*-\s*\d+(?:\s*-\s*[A-Za-zА-ЯЁ])?|\d+)\b', line)
    if num_match:
        result["scene_no"] = num_match.group(1).strip()
    
    type_match = re.search(r'\b(ИНТ\.?|НАТ\.?|INT\.?|EXT\.?|[иінат]+\.?)\b', line, re.IGNORECASE)
    if type_match:
        result["place_type"] = normalize_place_type(type_match.group(1))
    
    tod_match = re.search(r'\b(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО|день|ночь|вечер|утро|DAY|NIGHT)\b', line, re.IGNORECASE)
    if tod_match:
        result["tod"] = normalize_tod(tod_match.group(1))
        if result["place_type"] and result["tod"]:
            loc_pattern = rf'{re.escape(result["place_type"])}\s*(.+?)\s*[-–—:]\s*{re.escape(result["tod"])}'
            loc_match = re.search(loc_pattern, line, re.IGNORECASE)
            if loc_match:
                result["location"] = loc_match.group(1).strip().strip('.')
        elif result["tod"]:
            loc_match = re.search(r'(.+?)\s*[-–—:]\s*' + re.escape(result["tod"]), line, re.IGNORECASE)
            if loc_match:
                result["location"] = loc_match.group(1).strip().strip('.')
    
    return result

def parse_header(scene_text: str):
    lines = scene_text.splitlines()
    first_line = lines[0] if lines else scene_text[:200]
    
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
    
    return heuristic_parse(first_line)

# ===== IO helpers =====
CAST_LINE_RE = re.compile(r'^\s*\[.*?\]\s*$', re.MULTILINE)
UNDERLINE_MARK_RE = re.compile(r'\{\.underline\}', re.IGNORECASE)
BOLD_MARK_RE = re.compile(r'\*\*(.*?)\*\*')
LINE_BACKSLASH_RE = re.compile(r'\\\s*$')
EP_RE = re.compile(r'\[\s*ep\s*:\s*([^\]]+)\]', re.IGNORECASE)

def read_pdf(path):
    txt = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            txt += t + "\n"
    return txt

def read_docx(path):
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "")
        t = UNDERLINE_MARK_RE.sub('', t).replace('{.smallcaps}', '')
        t = BOLD_MARK_RE.sub(r'\1', t)
        t = LINE_BACKSLASH_RE.sub('', t)
        parts.append(t)
    txt = "\n".join(parts)
    txt = CAST_LINE_RE.sub('', txt)
    txt = re.sub(r'[ \t]+\n', '\n', txt)
    txt = txt.replace("\\[", "[").replace("\\]", "]")
    txt = re.sub(r"[ \t]*\\\\\s*$", "", txt, flags=re.MULTILINE)
    return txt

def read_script(path):
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    elif path.lower().endswith(".docx"):
        return read_docx(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

# ===== Rule-based keywords =====
def load_keywords(folder="keywords"):
    cats = ["violence", "sexual", "profanity", "alcohol_drugs", "scary"]
    keywords = {}
    weights = {}
    for cat in cats:
        path = os.path.join(folder, f"{cat}.txt")
        words, w = [], {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" in line:
                        word, weight = line.split(":", 1)
                        word = word.strip()
                        weight = float(weight.strip())
                    else:
                        word, weight = line, 1.0
                    words.append(word)
                    w[word] = weight
        keywords[cat] = words
        weights[cat] = w
    return keywords, weights

keywords, keyword_weights = load_keywords()

def find_triggers_weighted(text, words, weights):
    hits = []
    low = text.lower()
    total_score = 0.0
    for w in words:
        weight = weights.get(w, 1.0)
        for m in re.finditer(rf'\b{re.escape(w)}\b', low):
            start = max(0, m.start() - 25)
            end = min(len(text), m.end() + 25)
            snippet = text[start:end].replace("\n", " ")
            hits.append({"offset": m.start(), "match": w, "weight": weight, "snippet": snippet})
            total_score += weight
    return hits, total_score

def rule_based_score(scene_text):
    text = scene_text[:8000]
    result = {k: 0.0 for k in keywords}
    episodes = {k: [] for k in keywords}
    for cat, words in keywords.items():
        if not words:
            continue
        trig, total = find_triggers_weighted(text, words, keyword_weights[cat])
        episodes[cat].extend(trig)
        score = min(1.0, np.log1p(total) * 0.25)
        result[cat] = score
    return result, episodes

# ===== Manual ep features =====
MAP_KEY = {"v": "violence", "p": "profanity", "s": "sexual", "a": "alcohol_drugs", "sc": "scary"}
SEV_TO_NUM = {"None": 0.0, "Mild": 0.33, "Moderate": 0.66, "Severe": 1.0}

def parse_ep_features(text):
    max_sev = {k: 0.0 for k in keywords}
    count = {k: 0 for k in keywords}
    for m in EP_RE.finditer(text):
        payload = m.group(1)
        fields = {}
        for part in [x.strip() for x in payload.split(",") if x.strip()]:
            if "=" in part:
                k, v = [t.strip() for t in part.split("=", 1)]
                fields[k.lower()] = v
        for short, full in MAP_KEY.items():
            if short in fields:
                sev_val = SEV_TO_NUM.get(fields[short].title(), 0.66)
                max_sev[full] = max(max_sev[full], sev_val)
                count[full] += 1
        if "cat" in fields:
            full = MAP_KEY.get(fields["cat"].lower(), fields["cat"].lower())
            sev = fields.get("sev", "Moderate").title()
            sev_val = SEV_TO_NUM.get(sev, 0.66)
            if full in max_sev:
                max_sev[full] = max(max_sev[full], sev_val)
                count[full] += 1
    cats = list(keywords.keys())
    vec = [max_sev[c] for c in cats] + [count[c] for c in cats]
    return vec

# ===== ML Model =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL = "ai-forever/ruRoberta-large"
tok = AutoTokenizer.from_pretrained(EMB_MODEL)
mdl = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE)
mdl.eval()

def rule_vec(text):
    lf = text.lower()
    return np.array([sum(len(re.findall(rf'\b{re.escape(w)}\b', lf)) for w in keywords.get(cat, []))
                     for cat in ["violence", "sexual", "profanity", "alcohol_drugs", "scary"]], dtype=float)

# ===== Episode aggregates =====
from episodes_aggregates import episode_aggregates_for_scene

# ===== Load scene heads =====
if os.path.exists("heads.pkl"):
    with open("heads.pkl", "rb") as f:
        HEADS = pickle.load(f)
else:
    HEADS = None

# ===== Legal overrides (436-ФЗ) =====
OBSCENE_PATTERNS = [
    r"\b(еб[ао]н|пизд|хуй|охуенн|бляд)\w*\b",
    r"\b(нахуй|пидор|сука)\b",
]

DRUG_INSTRUCTIVE = [
    r"\b(как (приготовить|сделать|варить|закладк)|дозир\w+|инструкц\w+)\b",
]

DRUG_ROMANTICIZE = [
    r"\b(кайф|оргазм от|классно|без последств|ничего страшного)\b",
]

NATURALISTIC_VIOLENCE = [
    r"\b(натуралистическ\w+|крупным планом|в деталях)\b",
    r"\b(вскрыл|внутренност|кишк|кровищ)\w*\b",
]

EXPLICIT_SEX = [
    r"\b(половой акт|вводит|фрикц\w+|эякуляц\w+|оральн\w+)\b",
]

def any_match(text, patterns):
    low = text.lower()
    return any(re.search(p, low) for p in patterns)

def legal_overrides(scene_text):
    reasons = []
    if any_match(scene_text, OBSCENE_PATTERNS):
        reasons.append("Обсценная лексика (436‑ФЗ)")
        return {"min_age": "18+", "reasons": reasons}
    if any_match(scene_text, DRUG_INSTRUCTIVE) or any_match(scene_text, DRUG_ROMANTICIZE):
        reasons.append("Наркотики: инструктивность/романтизация (436‑ФЗ)")
        return {"min_age": "18+", "reasons": reasons}
    if any_match(scene_text, NATURALISTIC_VIOLENCE):
        reasons.append("Натуралистическое насилие (436‑ФЗ)")
        return {"min_age": "18+", "reasons": reasons}
    if any_match(scene_text, EXPLICIT_SEX):
        reasons.append("Детализированное сексуальное описание (436‑ФЗ)")
        return {"min_age": "18+", "reasons": reasons}
    return None

# ===== Thresholds and severity =====
THRESH = {"None": 0.2, "Mild": 0.4, "Moderate": 0.7}

def to_severity(p):
    if p < THRESH["None"]:
        return "None"
    if p < THRESH["Mild"]:
        return "Mild"
    if p < THRESH["Moderate"]:
        return "Moderate"
    return "Severe"

def analyze_scene(scene_text):
    rule_scores, episodes = rule_based_score(scene_text)
    ep_feats_vec = parse_ep_features(scene_text)
    epi = episode_aggregates_for_scene(scene_text)
    emb = scene_vector(scene_text, max_len=384, stride=320, batch_size=8, use_cache=True)
    rv = rule_vec(scene_text)
    
    if HEADS:
        x = np.hstack([emb, rv, ep_feats_vec, epi])
        model_probs = {cat: float(clf.predict_proba([x])[0, 1]) for cat, clf in HEADS.items()}
        epi_cat_max = {c: float(epi[i * 6 + 0]) for i, c in enumerate(["violence", "sexual", "profanity", "alcohol_drugs", "scary"])}
        final_probs = {cat: 0.55 * model_probs[cat] + 0.25 * rule_scores[cat] + 0.20 * epi_cat_max[cat]
                       for cat in ["violence", "sexual", "profanity", "alcohol_drugs", "scary"]}
    else:
        model_probs = {c: 0.0 for c in ["violence", "sexual", "profanity", "alcohol_drugs", "scary"]}
        epi_cat_max = {c: float(epi[i * 6 + 0]) for i, c in enumerate(["violence", "sexual", "profanity", "alcohol_drugs", "scary"])}
        final_probs = {cat: 0.80 * rule_scores[cat] + 0.20 * epi_cat_max[cat]
                       for cat in ["violence", "sexual", "profanity", "alcohol_drugs", "scary"]}
    
    severity = {cat: to_severity(p) for cat, p in final_probs.items()}
    per_class = {cat: {
        "rule_score": float(rule_scores[cat]),
        "model_proba": float(model_probs.get(cat, 0.0)),
        "episode_max": float(epi_cat_max[cat]),
        "final_proba": float(final_probs[cat]),
        "severity": severity[cat],
        "episodes": episodes[cat]
    } for cat in ["violence", "sexual", "profanity", "alcohol_drugs", "scary"]}
    
    return per_class

# ===== Age Rating =====
def age_from_scene(per_class):
    if per_class["profanity"]["severity"] in ["Moderate", "Severe"]:
        return "18+"
    if per_class["sexual"]["severity"] == "Severe":
        return "18+"
    if per_class["violence"]["severity"] == "Severe":
        return "18+"
    if per_class["violence"]["severity"] == "Moderate" or per_class["sexual"]["severity"] == "Moderate":
        return "16+"
    if per_class["alcohol_drugs"]["severity"] in ["Moderate", "Severe"]:
        return "16+"
    if per_class["scary"]["severity"] in ["Mild", "Moderate"]:
        return "12+"
    return "6+"

def aggregate_rating(scene_levels):
    order = ["0+", "6+", "12+", "16+", "18+"]
    worst = "0+"
    for r in scene_levels:
        if order.index(r) > order.index(worst):
            worst = r
    return worst

# ===== Main =====
def analyze_script(path, report_path="final_report.json"):
    text = read_script(path)
    text = normalize_headings(text)
    scenes = split_scenes(text)
    details, scene_levels = [], []
    
    print(f"Найдено сцен: {len(scenes)}")
    
    for i, s in enumerate(scenes, 1):
        meta = parse_header(s)
        per_class = analyze_scene(s)
        scene_rate = age_from_scene(per_class)
        
        override = legal_overrides(s)
        if override:
            order = ["0+", "6+", "12+", "16+", "18+"]
            if order.index(override["min_age"]) > order.index(scene_rate):
                scene_rate = override["min_age"]
        
        scene_levels.append(scene_rate)
        
        problems = []
        for cat, data in per_class.items():
            if data["severity"] in ["Moderate", "Severe"]:
                for ep in data["episodes"][:5]:
                    problems.append({
                        "category": cat,
                        "severity": data["severity"],
                        "snippet": ep["snippet"],
                        "offset": ep["offset"]
                    })
        
        if override:
            for r in override["reasons"]:
                problems.append({
                    "category": "legal",
                    "severity": "Severe",
                    "snippet": s[:240],
                    "offset": 0
                })
        
        details.append({
            "scene_index": i,
            **meta,
            "per_class": {k: {
                "rule_score": data["rule_score"],
                "model_proba": data["model_proba"],
                "episode_max": data["episode_max"],
                "final_proba": data["final_proba"],
                "severity": data["severity"],
                "episodes_count": len(data["episodes"])
            } for k, data in per_class.items()},
            "scene_rating": scene_rate,
            "problems": problems
        })
        
        print(f"Сцена {i}: {scene_rate} | {meta.get('scene_no', '')} {meta.get('place_type', '')} {meta.get('location', '')} - {meta.get('tod', '')}")
    
    rating = aggregate_rating(scene_levels)
    
    def pct(cat):
        cnt = sum(1 for d in details if d["per_class"][cat]["severity"] in ["Mild", "Moderate", "Severe"])
        return round(100.0 * cnt / max(1, len(details)), 2)
    
    guide = {
        "violence": {"percentage_scenes": pct("violence"), "episodes_total": sum(d["per_class"]["violence"]["episodes_count"] for d in details)},
        "sexual": {"percentage_scenes": pct("sexual"), "episodes_total": sum(d["per_class"]["sexual"]["episodes_count"] for d in details)},
        "profanity": {"percentage_scenes": pct("profanity"), "episodes_total": sum(d["per_class"]["profanity"]["episodes_count"] for d in details)},
        "alcohol_drugs": {"percentage_scenes": pct("alcohol_drugs"), "episodes_total": sum(d["per_class"]["alcohol_drugs"]["episodes_count"] for d in details)},
        "scary": {"percentage_scenes": pct("scary"), "episodes_total": sum(d["per_class"]["scary"]["episodes_count"] for d in details)},
    }
    
    payload = {
        "rating": rating,
        "summary": {
            "count_scenes": len(scenes),
            "scene_ratings": {r: scene_levels.count(r) for r in ["6+", "12+", "16+", "18+"]}
        },
        "parents_guide": guide,
        "details": details
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Итоговый рейтинг: {rating}")
    print(f"📁 Сохранён отчёт: {report_path}")

if __name__ == "__main__":
    path = input("Введите путь к сценарию (.docx/.pdf): ").strip()
    if not os.path.exists(path):
        print("Файл не найден.")
    else:
        analyze_script(path)
