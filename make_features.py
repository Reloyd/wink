# make_features.py — сбор признаков под test.py (mvp_rating)
import numpy as np
import pandas as pd
import torch
import re

from transformers import AutoTokenizer, AutoModel

# Берём те же помощники и словари, что в test.py
from test import read_script, split_scenes, keywords
from episodes_aggregates import episode_aggregates_for_scene

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "ai-forever/ruRoberta-large"

EP_RE = re.compile(r'\[\s*ep\s*:\s*([^\]]+)\]', re.IGNORECASE)
MAP_KEY = {"v":"violence","p":"profanity","s":"sexual","a":"alcohol_drugs","sc":"scary"}
SEV_TO_NUM = {"None":0.0,"Mild":0.33,"Moderate":0.66,"Severe":1.0}

def parse_ep_features(text):
    """[ep: ...] → 10 признаков: 5 max_sev(0..1) + 5 count"""
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
    cats = list(keywords.keys())
    return [max_sev[c] for c in cats] + [count[c] for c in cats]

def embed_mean(texts):
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(DEVICE)
    mdl.eval()
    out = []
    with torch.no_grad():
        for t in texts:
            x = tok(t[:2000], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            hs = mdl(**x).last_hidden_state         # [1, T, H]
            attn = (x["attention_mask"].unsqueeze(-1) > 0).float()
            v = (hs * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)  # mean-pool
            out.append(v.cpu().numpy())
    return np.vstack(out)  # [N, 1024] для ruRoberta-large

def rule_feats(text):
    lf = text.lower()
    feats = []
    # порядок категорий должен совпадать с test.py
    for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]:
        words = keywords.get(cat, [])
        cnt = sum(len(re.findall(re.escape(w), lf)) for w in words)
        feats.append(cnt)
    return feats  # [5]

def run(script_path, labels_csv, out_prefix="data"):
    df = pd.read_csv(labels_csv)
    text = read_script(script_path)
    scenes = split_scenes(text)

    print("Эмбеддинги сцен...")
    embs = embed_mean([s for s in scenes])                 # [N, 1024]

    print("Правила и ручные ep-фичи...")
    rules = np.array([rule_feats(s) for s in scenes])      # [N, 5]
    ep_feats = np.array([parse_ep_features(s) for s in scenes])  # [N, 10]

    print("Агрегаты эпизодов...")
    epi = np.array([episode_aggregates_for_scene(s) for s in scenes])  # [N, 30]

    # Собираем X = [emb | rule | ep | epi] = [1024 + 5 + 10 + 30] = 1069
    X = np.hstack([embs, rules, ep_feats, epi])
    print("X shape:", X.shape)  # ожидаем (*, 1069)
    np.save(f"{out_prefix}_X.npy", X)

    # Цели по категориям (как раньше)
    for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]:
        y = df[f"has_{cat}"].values
        np.save(f"{out_prefix}_y_{cat}.npy", y)

    print(f"✅ Сохранено: {out_prefix}_X.npy, {out_prefix}_y_*.npy")

if __name__ == "__main__":
    run("annotated_script.docx","labels.csv")
