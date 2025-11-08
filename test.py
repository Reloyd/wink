# mvp_rating.py — инференс с эпизодными агрегатами
import os, re, json, pickle
import numpy as np
import torch
import pdfplumber
from docx import Document
from transformers import AutoTokenizer, AutoModel

# ===== Regex & Parsing =====
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

# ===== IO helpers =====
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
                    if not line or line.startswith("#"): continue
                    if ":" in line:
                        word, weight = line.split(":", 1)
                        word = word.strip(); weight = float(weight.strip())
                    else:
                        word, weight = line, 1.0
                    words.append(word); w[word] = weight
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
        for m in re.finditer(re.escape(w), low):
            start = max(0, m.start()-25)
            end = min(len(text), m.end()+25)
            snippet = text[start:end].replace("\n"," ")
            hits.append({"offset": m.start(), "match": w, "weight": weight, "snippet": snippet})
            total_score += weight
    return hits, total_score

def rule_based_score(scene_text):
    text = scene_text[:8000]
    result = {k: 0.0 for k in keywords}
    episodes = {k: [] for k in keywords}
    for cat, words in keywords.items():
        if not words: continue
        trig, total = find_triggers_weighted(text, words, keyword_weights[cat])
        episodes[cat].extend(trig)
        score = min(1.0, np.log1p(total) * 0.25)
        result[cat] = score
    return result, episodes

# ===== Manual ep features from [ep: ...] =====
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

# ===== ML Model (scene encoder) =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL = "ai-forever/ruRoberta-large"
tok = AutoTokenizer.from_pretrained(EMB_MODEL)
mdl = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE)
mdl.eval()

def embed_scene_mean(text):
    x = tok(text[:2000], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        hs = mdl(**x).last_hidden_state
        attn = (x["attention_mask"].unsqueeze(-1) > 0).float()
        vec = (hs * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
        return vec.cpu().numpy().ravel()

def rule_vec(text):
    lf = text.lower()
    return np.array([sum(lf.count(w) for w in keywords.get(cat, [])) for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]], dtype=float)

# ===== Episode aggregates (import trained heads) =====
from episodes_aggregates import episode_aggregates_for_scene

# ===== Load scene heads =====
if os.path.exists("heads.pkl"):
    with open("heads.pkl","rb") as f:
        HEADS = pickle.load(f)
else:
    HEADS = None

THRESH = {"None":0.2,"Mild":0.4,"Moderate":0.7}
def to_severity(p):
    if p < THRESH["None"]: return "None"
    if p < THRESH["Mild"]: return "Mild"
    if p < THRESH["Moderate"]: return "Moderate"
    return "Severe"

def analyze_scene(scene_text):
    rule_scores, episodes = rule_based_score(scene_text)
    ep_feats_vec = parse_ep_features(scene_text)            # 10 фич (manual)
    epi = episode_aggregates_for_scene(scene_text)          # 30 фич (episode heads)
    emb = embed_scene_mean(scene_text)
    rv = rule_vec(scene_text)

    if HEADS:
        x = np.hstack([emb, rv, ep_feats_vec, epi])

        model_probs = {cat: float(clf.predict_proba([x])[0,1]) for cat, clf in HEADS.items()}
        # Эпизодный вклад по категории: берём для простоты первый агрегат (max p_bin)
        epi_cat_max = {c: float(epi[i*6 + 0]) for i, c in enumerate(["violence","sexual","profanity","alcohol_drugs","scary"])}
        final_probs = {cat: 0.55*model_probs[cat] + 0.25*rule_scores[cat] + 0.20*epi_cat_max[cat]
                       for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]}
    else:
        model_probs = {c: 0.0 for c in ["violence","sexual","profanity","alcohol_drugs","scary"]}
        epi_cat_max = {c: float(epi[i*6 + 0]) for i, c in enumerate(["violence","sexual","profanity","alcohol_drugs","scary"])}
        final_probs = {cat: 0.80*rule_scores[cat] + 0.20*epi_cat_max[cat]
                       for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]}

    severity = {cat: to_severity(p) for cat, p in final_probs.items()}

    per_class = {cat: {
        "rule_score": float(rule_scores[cat]),
        "model_proba": float(model_probs.get(cat, 0.0)),
        "episode_max": float(epi_cat_max[cat]),
        "final_proba": float(final_probs[cat]),
        "severity": severity[cat],
        "episodes": episodes[cat]
    } for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]}

    return per_class

# ===== Age Rating =====
def age_from_scene(per_class):
    if per_class["profanity"]["severity"] in ["Moderate","Severe"]: return "18+"
    if per_class["sexual"]["severity"] == "Severe": return "18+"
    if per_class["violence"]["severity"] == "Severe": return "18+"
    if per_class["violence"]["severity"] == "Moderate" or per_class["sexual"]["severity"] == "Moderate": return "16+"
    if per_class["alcohol_drugs"]["severity"] in ["Moderate","Severe"]: return "16+"
    if per_class["scary"]["severity"] in ["Mild","Moderate"]: return "12+"
    return "6+"

def aggregate_rating(scene_levels):
    order = ["0+","6+","12+","16+","18+"]
    worst = "0+"
    for r in scene_levels:
        if order.index(r) > order.index(worst): worst = r
    return worst

# ===== Main =====
def analyze_script(path, report_path="final_report.json"):
    text = read_script(path)
    scenes = split_scenes(text)
    details, scene_levels = [], []

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
                "episode_max": data["episode_max"],
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
