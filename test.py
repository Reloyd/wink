import os, re, json
import numpy as np
import pdfplumber
from docx import Document

# ---------- Regex и очистка ----------
SCENE_HEADING_RE = re.compile(
    r"""^\s*((?:\d+\s*-\s*\d+(?:\s*-\s*[А-ЯA-Z])?)\.?\s*(?:\d{1,2}-Е\.)?\s*(?:ИНТ\.|НАТ\.|INT\.|EXT\.|ИНТ|НАТ|INT|EXT)?[^\n]*?(?:\s*-\s*|\s+)?(?:ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО)\s*\d*\.?)\s*$""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE
)
HEADER_PARSE_RE = re.compile(
    r'^\s*(\d+\s*-\s*\d+(?:\s*-\s*[А-ЯA-Z])?)\.?\s*(\d{1,2}-Е\.)?\s*(ИНТ\.|НАТ\.|INT\.|EXT\.)?\s*([^-\n]*?)\s*(?:-\s*|\s+)?(ДЕНЬ|НОЧЬ|ВЕЧЕР|УТРО)\s*\d*\.?\s*$',
    re.IGNORECASE
)
CAST_LINE_RE = re.compile(r'^\s*\[.*?\]\s*$', re.MULTILINE)
UNDERLINE_MARK_RE = re.compile(r'\{\.underline\}', re.IGNORECASE)
BOLD_MARK_RE = re.compile(r'\*\*(.*?)\*\*')
LINE_BACKSLASH_RE = re.compile(r'\\\s*$')

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
    return txt

def read_script(path):
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    elif path.lower().endswith(".docx"):
        return read_docx(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def split_scenes(text):
    matches = list(SCENE_HEADING_RE.finditer(text))
    if not matches:
        parts = re.split(r'(?=^\s*(?:\d+\s*-\s*\d+\.|СЦЕНА\s*\d*\.|INT\.|EXT\.|ИНТ\.|НАТ\.))',
                         text, flags=re.IGNORECASE | re.MULTILINE)
        return [p.strip() for p in parts if len(p.split()) > 3]
    scenes = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if len(chunk.split()) > 3:
            scenes.append(chunk)
    return scenes

def parse_header(scene_text):
    first = scene_text.splitlines()[0] if scene_text.splitlines() else scene_text[:120]
    m = HEADER_PARSE_RE.search(first)
    if not m:
        return {"scene_no":"", "period":"", "place_type":"", "location":"", "tod":""}
    return {
        "scene_no": (m.group(1) or "").strip(),
        "period": (m.group(2) or "").strip(),
        "place_type": (m.group(3) or "").strip(),
        "location": (m.group(4) or "").strip().strip('. '),
        "tod": (m.group(5) or "").strip()
    }

# ---------- Словари и правила ----------
keywords = {
    "violence": ["удар", "ударил", "столкновени", "авар", "дтп", "скрежет", "кровь", "нож", "пистолет", "стрел", "труп", "убил"],
    "sexual": ["поцелу", "раздел", "постель", "эрот", "интим", "секс"],
    "profanity": ["нахрен", "охрен", "черт", "блин", "сука", "падла", "дерьмо", "хер", "еб"],
    "alcohol_drugs": ["водка", "бутылк", "пья", "виски", "алкогол", "спирт", "косяк", "курит", "наркотик"],
    "scary": ["страшн", "монстр", "крик", "крики", "тень", "пугает", "ужас"]
}

def find_triggers(text, words):
    hits = []
    low = text.lower()
    for w in words:
        for m in re.finditer(re.escape(w), low):
            start = max(0, m.start()-25); end = min(len(text), m.end()+25)
            snippet = text[start:end].replace("\n"," ")
            hits.append({"offset": m.start(), "match": w, "snippet": snippet})
    return hits

def analyze_scene(scene_text):
    text = scene_text[:8000]
    low = text.lower()
    result = {k: 0.0 for k in keywords}
    episodes = {k: [] for k in keywords}

    for cat, words in keywords.items():
        trig = find_triggers(text, words)
        episodes[cat].extend(trig)
        score = min(1.0, np.log1p(len(trig)) * 0.35)
        result[cat] = score

    # Усилители
    if re.search(r'\bводка|бутылк|пья|виски|алкогол|спирт\b', low):
        result["alcohol_drugs"] = min(1.0, result["alcohol_drugs"] + 0.4)
    if re.search(r'\bстолкновени|авар|дтп|скрежет|кровь|труп|удар\b', low):
        result["violence"] = min(1.0, result["violence"] + 0.3)
    if re.search(r'\bстрашн|ужас|крик|крики|пугает|монстр|тень\b', low):
        result["scary"] = min(1.0, result["scary"] + 0.25)
    if re.search(r'\bнахрен|охрен|сука|хер|блин|дерьмо|еб', low):
        result["profanity"] = min(1.0, result["profanity"] + 0.35)

    # Severity маппинг
    def sev(x):
        if x < 0.2: return "None"
        if x < 0.4: return "Mild"
        if x < 0.7: return "Moderate"
        return "Severe"

    per_class = {k: {"score": float(v), "severity": sev(v), "episodes": episodes[k]} for k, v in result.items()}
    return per_class

# ---------- Рейтинг по 436-ФЗ-подобной логике ----------
def age_from_scene(per_class):
    # Жёсткие правила
    if per_class["profanity"]["severity"] in ["Moderate","Severe"]:
        return "18+"
    if per_class["sexual"]["severity"] == "Severe":
        return "18+"
    if per_class["violence"]["severity"] == "Severe":
        return "18+"
    # Средний уровень
    if per_class["violence"]["severity"] == "Moderate" or per_class["sexual"]["severity"] == "Moderate":
        return "16+"
    if per_class["alcohol_drugs"]["severity"] in ["Moderate","Severe"]:
        return "16+"
    # Мягкие
    if per_class["scary"]["severity"] in ["Mild","Moderate"]:
        return "12+"
    return "6+"

def aggregate_rating(scene_levels):
    # Самый тяжёлый элемент
    order = ["0+","6+","12+","16+","18+"]
    worst = "0+"
    for r in scene_levels:
        if order.index(r) > order.index(worst):
            worst = r
    return worst

# ---------- Основной поток ----------
def analyze_script(path, report_path="final_report.json"):
    text = read_script(path)
    scenes = split_scenes(text)
    details = []
    scene_levels = []

    for i, s in enumerate(scenes, 1):
        meta = parse_header(s)
        per_class = analyze_scene(s)
        scene_rate = age_from_scene(per_class)
        scene_levels.append(scene_rate)

        # собрать проблемные эпизоды
        problems = []
        for cat, data in per_class.items():
            if data["severity"] in ["Moderate","Severe"]:
                for ep in data["episodes"][:5]:  # ограничим
                    problems.append({
                        "category": cat,
                        "severity": data["severity"],
                        "snippet": ep["snippet"],
                        "offset": ep["offset"]
                    })

        details.append({
            "scene_index": i,
            **meta,
            "per_class": {k: {"score": data["score"], "severity": data["severity"], "episodes": len(data["episodes"])} for k, data in per_class.items()},
            "scene_rating": scene_rate,
            "problems": problems
        })
        print(f"Сцена {i}: рейтинг {scene_rate}; {meta.get('scene_no','')} {meta.get('place_type','')} {meta.get('location','')} - {meta.get('tod','')}")

    rating = aggregate_rating(scene_levels)

    # Статистика для Parents Guide
    def pct(cat):
        cnt = sum(1 for d in details if d["per_class"][cat]["severity"] in ["Mild","Moderate","Severe"])
        return round(100.0 * cnt / max(1, len(details)), 2)

    guide = {
        "violence": {"percentage_scenes": pct("violence"), "episodes_total": sum(d["per_class"]["violence"]["episodes"] for d in details)},
        "sexual": {"percentage_scenes": pct("sexual"), "episodes_total": sum(d["per_class"]["sexual"]["episodes"] for d in details)},
        "profanity": {"percentage_scenes": pct("profanity"), "episodes_total": sum(d["per_class"]["profanity"]["episodes"] for d in details)},
        "alcohol_drugs": {"percentage_scenes": pct("alcohol_drugs"), "episodes_total": sum(d["per_class"]["alcohol_drugs"]["episodes"] for d in details)},
        "scary": {"percentage_scenes": pct("scary"), "episodes_total": sum(d["per_class"]["scary"]["episodes"] for d in details)},
    }

    payload = {
        "rating": rating,
        "summary": {
            "count_scenes": len(scenes),
            "scene_ratings": {r: scene_levels.count(r) for r in ["6+","12+","16+","18+"]}
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
