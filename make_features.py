# make_features.py
import numpy as np, pandas as pd, torch, re
from transformers import AutoTokenizer, AutoModel
from test import read_script, split_scenes, keywords

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "ai-forever/ruRoberta-large"

EP_RE = re.compile(r'\[\s*ep\s*:\s*([^\]]+)\]', re.IGNORECASE)
MAP_KEY = {"v":"violence","p":"profanity","s":"sexual","a":"alcohol_drugs","sc":"scary"}
SEV_TO_NUM = {"None":0.0,"Mild":0.33,"Moderate":0.66,"Severe":1.0}

def parse_ep_features(text):
    """Извлечь из [ep: ...] тегов максимальную серьёзность и количество по категориям"""
    max_sev = {k: 0.0 for k in keywords}
    count = {k: 0 for k in keywords}
    
    for m in EP_RE.finditer(text):
        payload = m.group(1)
        fields = {}
        for part in [x.strip() for x in payload.split(",") if x.strip()]:
            if "=" in part:
                k, v = [t.strip() for t in part.split("=", 1)]
                fields[k.lower()] = v
        
        # короткие ключи v,p,s,a,sc
        for short, full in MAP_KEY.items():
            if short in fields:
                sev_val = SEV_TO_NUM.get(fields[short].title(), 0.66)
                max_sev[full] = max(max_sev[full], sev_val)
                count[full] += 1
        
        # длинные ключи cat=..., sev=...
        if "cat" in fields:
            full = MAP_KEY.get(fields["cat"].lower(), fields["cat"].lower())
            sev = fields.get("sev","Moderate").title()
            sev_val = SEV_TO_NUM.get(sev, 0.66)
            if full in max_sev:
                max_sev[full] = max(max_sev[full], sev_val)
                count[full] += 1
    
    # Вернём вектор: [max_sev_violence, max_sev_sexual, ..., count_violence, count_sexual, ...]
    cats = list(keywords.keys())
    vec = [max_sev[c] for c in cats] + [count[c] for c in cats]
    return vec

def encode(texts):
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(DEVICE)
    mdl.eval()
    out = []
    with torch.no_grad():
        for t in texts:
            t = t[:2000]
            x = tok(t, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            h = mdl(**x).last_hidden_state[:,0,:].cpu().numpy()
            out.append(h)
    return np.vstack(out)

def rule_feats(text):
    lf = text.lower()
    feats = []
    for cat, words in keywords.items():
        cnt = sum(len(re.findall(re.escape(w), lf)) for w in words)
        feats.append(cnt)
    return feats

def run(script_path, labels_csv, out_prefix="data"):
    df = pd.read_csv(labels_csv)
    text = read_script(script_path)
    scenes = split_scenes(text)
    
    print("Получение эмбеддингов...")
    embs = encode([s for s in scenes])
    
    print("Извлечение rule-фич и эпизодных признаков...")
    rules = np.array([rule_feats(s) for s in scenes])
    ep_feats = np.array([parse_ep_features(s) for s in scenes])
    
    # Объединяем: [emb | rule | ep]
    X = np.hstack([embs, rules, ep_feats])
    np.save(f"{out_prefix}_X.npy", X)
    
    cats = ["violence","sexual","profanity","alcohol_drugs","scary"]
    for cat in cats:
        y = df[f"has_{cat}"].values
        np.save(f"{out_prefix}_y_{cat}.npy", y)
    
    print(f"✅ Признаки сохранены: {out_prefix}_X.npy ({X.shape}), {out_prefix}_y_*.npy")

if __name__ == "__main__":
    run("annotated_script.docx","labels.csv")
