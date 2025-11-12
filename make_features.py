# make_features.py — сбор признаков под test.py (оконный эмбеддинг сцен)

import numpy as np
import pandas as pd
import torch
import re

# Общий оконный эмбеддинг сцен
from embeddings import scene_vector  # 3H агрегат: mean | max | top3-mean [web:35][web:39]

# Берём те же помощники и словари, что в test.py
from test import read_script, split_scenes, keywords
from episodes_aggregates import episode_aggregates_for_scene  # 30 фич (эпизодные головы) [web:6]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EP_RE = re.compile(r'\[\s*ep\s*:\s*([^\]]+)\]', re.IGNORECASE)
MAP_KEY = {"v":"violence","p":"profanity","s":"sexual","a":"alcohol_drugs","sc":"scary"}
SEV_TO_NUM = {"None":0.0,"Mild":0.33,"Moderate":0.66,"Severe":1.0}

def parse_ep_features(text):
    """[ep: ...] → 10 признаков: 5 max_sev(0..1) + 5 count (как в test.py)"""
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
    return [max_sev[c] for c in cats] + [count[c] for c in cats]  # [10] [web:6]

def rule_feats(text):
    """Подсчет совпадений словарей по категориям (границы слова, как в test.py)"""
    lf = text.lower()
    feats = []
    for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]:
        words = keywords.get(cat, [])
        cnt = sum(len(re.findall(rf'\b{re.escape(w)}\b', lf)) for w in words)
        feats.append(cnt)
    return feats  # [5] [web:39]

def run(script_path, labels_csv, out_prefix="data"):
    # 1) Читаем метки и сцены
    df = pd.read_csv(labels_csv)
    text = read_script(script_path)
    scenes = split_scenes(text)

    # 2) Эмбеддинги сцен (оконный агрегат 3H)
    # Используем кэш внутри scene_vector(use_cache=True) для скорости повторных запусков
    print("Эмбеддинги сцен (sliding windows)...")
    embs = np.vstack([
        scene_vector(s, max_len=384, stride=320, batch_size=8, use_cache=True) for s in scenes
    ])  # [N, 3H] [web:35][web:39]

    # 3) Правила и ручные ep-фичи
    print("Правила и ручные ep-фичи...")
    rules = np.array([rule_feats(s) for s in scenes], dtype=float)           # [N, 5] [web:39]
    ep_feats = np.array([parse_ep_features(s) for s in scenes], dtype=float) # [N, 10] [web:6]

    # 4) Агрегаты эпизодов (на токенных окнах)
    print("Агрегаты эпизодов...")
    epi = np.array([episode_aggregates_for_scene(s) for s in scenes], dtype=float)  # [N, 30] [web:6]

    # 5) Собираем X = [scene_emb(3H) | rule(5) | ep(10) | epi(30)]
    X = np.hstack([embs, rules, ep_feats, epi])
    print("X shape:", X.shape)
    np.save(f"{out_prefix}_X.npy", X)

    # 6) Цели по категориям
    for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]:
        y = df[f"has_{cat}"].values
        np.save(f"{out_prefix}_y_{cat}.npy", y)

    print(f"✅ Сохранено: {out_prefix}_X.npy, {out_prefix}_y_*.npy")

if __name__ == "__main__":
    # Пример: python make_features.py
    # Ожидает, что есть annotated_script.docx и labels.csv из extract_labels.py
    run("annotated_script.docx","labels.csv")
