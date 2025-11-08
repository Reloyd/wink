# extract_episodes.py (with severity)
import re, csv, sys
from docx import Document

EP_RE = re.compile(r'\[\s*ep\s*:\s*([^\]]+)\]', re.IGNORECASE)
MAP_KEY = {"v":"violence","p":"profanity","s":"sexual","a":"alcohol_drugs","sc":"scary"}
SEV_NORM = {"none":"None","mild":"Mild","moderate":"Moderate","severe":"Severe",
            "нет":"None","лёгкое":"Mild","легкое":"Mild","среднее":"Moderate","жёсткое":"Severe","жесткое":"Severe"}
SEV_TO_NUM = {"None":0.0, "Mild":0.33, "Moderate":0.66, "Severe":1.0}

def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text or "" for p in doc.paragraphs)

def window_around(text, idx, radius=220):
    start = max(0, idx - radius)
    end = min(len(text), idx + radius)
    return text[start:end].strip()

def normalize(text: str) -> str:
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = re.sub(r"[ \t]*\\\\\s*$", "", text, flags=re.MULTILINE)
    return text

def extract(path, out_csv="episodes.csv"):
    raw = read_docx(path) if path.lower().endswith(".docx") else open(path, encoding="utf-8").read()
    text = normalize(raw)
    rows = []
    for m in EP_RE.finditer(text):
        payload = m.group(1)
        fields = {}
        for part in [x.strip() for x in payload.split(",") if x.strip()]:
            if "=" in part:
                k, v = [t.strip() for t in part.split("=", 1)]
                fields[k.lower()] = v

        # инициализация
        bin_labels = {k:0 for k in ["violence","sexual","profanity","alcohol_drugs","scary"]}
        sev_labels = {k:"None" for k in ["violence","sexual","profanity","alcohol_drugs","scary"]}

        # короткие ключи v,p,s,a,sc с уровнем
        for short, full in MAP_KEY.items():
            if short in fields:
                sev_txt = SEV_NORM.get(fields[short].lower(), fields[short].title())
                bin_labels[full] = 1
                sev_labels[full] = sev_txt

        # длинные ключи cat=..., sev=...
        if "cat" in fields:
            full = MAP_KEY.get(fields["cat"].lower(), fields["cat"].lower())
            sev_txt = SEV_NORM.get(fields.get("sev","Moderate").lower(), fields.get("sev","Moderate").title())
            if full in bin_labels:
                bin_labels[full] = 1
                sev_labels[full] = sev_txt

        snippet = window_around(text, m.start(), 200)

        row = {
            "text": snippet,
            **bin_labels,
            "sev_violence": sev_labels["violence"],
            "sev_sexual": sev_labels["sexual"],
            "sev_profanity": sev_labels["profanity"],
            "sev_alcohol_drugs": sev_labels["alcohol_drugs"],
            "sev_scary": sev_labels["scary"],
            "sev_violence_num": SEV_TO_NUM[sev_labels["violence"]],
            "sev_sexual_num": SEV_TO_NUM[sev_labels["sexual"]],
            "sev_profanity_num": SEV_TO_NUM[sev_labels["profanity"]],
            "sev_alcohol_drugs_num": SEV_TO_NUM[sev_labels["alcohol_drugs"]],
            "sev_scary_num": SEV_TO_NUM[sev_labels["scary"]],
        }
        rows.append(row)

    fields = ["text","violence","sexual","profanity","alcohol_drugs","scary",
              "sev_violence","sev_sexual","sev_profanity","sev_alcohol_drugs","sev_scary",
              "sev_violence_num","sev_sexual_num","sev_profanity_num","sev_alcohol_drugs_num","sev_scary_num"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"✅ episodes with severity: {len(rows)} → {out_csv}")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv)>1 else "annotated_script.docx"
    extract(inp)
