# mvp_rating.py — полный код с эпизодными признаками
import os, re, json, pickle
import numpy as np
import torch
import pdfplumber
from docx import Document
from transformers import AutoTokenizer, AutoModel

# ============ Regex & Parsing ============
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

# ============ Rule-based keywords ============
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

def rule_based_score(scene_text):
    text = scene_text[:8000]
    low = text.lower()
    result = {k: 0.0 for k in keywords}
    episodes = {k: [] for k in keywords}

    for cat, words in keywords.items():
        trig = find_triggers(text, words)
        episodes[cat].extend(trig)
        score = min(1.0, np.log1p(len(trig)) * 0.35)
        result[cat] = score

    if re.search(r'\bводка|бутылк|пья|виски|алкогол|спирт\b', low):
        result["alcohol_drugs"] = min(1.0, result["alcohol_drugs"] + 0.4)
    if re.search(r'\bстолкновени|авар|дтп|скрежет|кровь|труп|удар\b', low):
        result["violence"] = min(1.0, result["violence"] + 0.3)
    if re.search(r'\bстрашн|ужас|крик|крики|пугает|монстр|тень\b', low):
        result["scary"] = min(1.0, result["scary"] + 0.25)
    if re.search(r'\bнахрен|охрен|сука|хер|блин|дерьмо|еб', low):
        result["profanity"] = min(1.0, result["profanity"] + 0.35)

    return result, episodes

# ============ Эпизодные признаки ============
MAP_KEY = {"v":"violence","p":"profanity","s":"sexual","a":"alcohol_drugs","sc":"scary"}
SEV_TO_NUM = {"None":0.0,"Mild":0.33,"Moderate":0.66,"Severe":1.0}

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
            sev = fields.get("sev","Moderate").title()
            sev_val = SEV_TO_NUM.get(sev, 0.66)
            if full in max_sev:
                max_sev[full] = max(max_sev[full], sev_val)
                count[full] += 1
    
    cats = list(keywords.keys())
    vec = [max_sev[c] for c in cats] + [count[c] for c in cats]
    return vec

# ============ ML Model ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL = "ai-forever/ruRoberta-large"

print("Загрузка модели RuRoBERTa...")
tok = AutoTokenizer.from_pretrained(EMB_MODEL)
mdl = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE)
mdl.eval()

if os.path.exists("heads.pkl"):
    with open("heads.pkl","rb") as f:
        HEADS = pickle.load(f)
    print("Обученные головы загружены из heads.pkl")
else:
    print("heads.pkl не найден, используется только rule-based")
    HEADS = None

def embed_scene(text):
    x = tok(text[:2000], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        h = mdl(**x).last_hidden_state[:,0,:].cpu().numpy().ravel()
    return h

def rule_vec(text):
    lf = text.lower()
    return np.array([sum(lf.count(w) for w in words) for words in keywords.values()], dtype=float)

THRESH = {"None":0.2,"Mild":0.4,"Moderate":0.7}

def to_severity(p):
    if p < THRESH["None"]: return "None"
    if p < THRESH["Mild"]: return "Mild"
    if p < THRESH["Moderate"]: return "Moderate"
    return "Severe"

def analyze_scene(scene_text):
    rule_scores, episodes = rule_based_score(scene_text)
    ep_feats_vec = parse_ep_features(scene_text)
    
    if HEADS:
        emb = embed_scene(scene_text)
        rv = rule_vec(scene_text)
        x = np.hstack([emb, rv, ep_feats_vec])
        model_probs = {cat: float(clf.predict_proba([x])[0,1]) for cat, clf in HEADS.items()}
        
        # Извлечём вклад эпизодов как среднее max_sev
        cats = list(keywords.keys())
        ep_max = {cats[i]: ep_feats_vec[i] for i in range(5)}
        
        # Гибрид: 55% модель, 30% правила, 15% ручные эпизоды
        final_probs = {cat: 0.55*model_probs[cat] + 0.30*rule_scores[cat] + 0.15*ep_max[cat] 
                       for cat in keywords}
    else:
        model_probs = {k: 0.0 for k in keywords}
        cats = list(keywords.keys())
        ep_max = {cats[i]: ep_feats_vec[i] for i in range(5)}
        final_probs = {cat: 0.85*rule_scores[cat] + 0.15*ep_max[cat] for cat in keywords}
    
    severity = {cat: to_severity(p) for cat, p in final_probs.items()}
    
    per_class = {cat: {
        "rule_score": float(rule_scores[cat]),
        "model_proba": float(model_probs.get(cat, 0.0)),
        "manual_ep_score": float(ep_max[cat]),
        "final_proba": float(final_probs[cat]),
        "severity": severity[cat],
        "episodes": episodes[cat]
    } for cat in keywords}
    
    return per_class

# ============ Age Rating ============
def age_from_scene(per_class):
    if per_class["profanity"]["severity"] in ["Moderate","Severe"]:
        return "18+"
    if per_class["sexual"]["severity"] == "Severe":
        return "18+"
    if per_class["violence"]["severity"] == "Severe":
        return "18+"
    if per_class["violence"]["severity"] == "Moderate" or per_class["sexual"]["severity"] == "Moderate":
        return "16+"
    if per_class["alcohol_drugs"]["severity"] in ["Moderate","Severe"]:
        return "16+"
    if per_class["scary"]["severity"] in ["Mild","Moderate"]:
        return "12+"
    return "6+"

def aggregate_rating(scene_levels):
    order = ["0+","6+","12+","16+","18+"]
    worst = "0+"
    for r in scene_levels:
        if order.index(r) > order.index(worst):
            worst = r
    return worst

# ============ Main Pipeline ============
def analyze_script(path, report_path="final_report.json"):
    text = read_script(path)
    scenes = split_scenes(text)
    details = []
    scene_levels = []

    print(f"Найдено сцен: {len(scenes)}")
    for i, s in enumerate(scenes, 1):
        meta = parse_header(s)
        per_class = analyze_scene(s)
        scene_rate = age_from_scene(per_class)
        scene_levels.append(scene_rate)

        problems = []
        for cat, data in per_class.items():
            if data["severity"] in ["Moderate","Severe"]:
                for ep in data["episodes"][:5]:
                    problems.append({
                        "category": cat,
                        "severity": data["severity"],
                        "snippet": ep["snippet"],
                        "offset": ep["offset"]
                    })

        details.append({
            "scene_index": i,
            **meta,
            "per_class": {k: {
                "rule_score": data["rule_score"],
                "model_proba": data["model_proba"],
                "manual_ep_score": data["manual_ep_score"],
                "final_proba": data["final_proba"],
                "severity": data["severity"],
                "episodes_count": len(data["episodes"])
            } for k, data in per_class.items()},
            "scene_rating": scene_rate,
            "problems": problems
        })
        print(f"Сцена {i}: {scene_rate} | {meta.get('scene_no','')} {meta.get('place_type','')} {meta.get('location','')} - {meta.get('tod','')}")

    rating = aggregate_rating(scene_levels)

    def pct(cat):
        cnt = sum(1 for d in details if d["per_class"][cat]["severity"] in ["Mild","Moderate","Severe"])
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
