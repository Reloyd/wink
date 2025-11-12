# episodes_aggregates.py — эпизодные агрегаты по токенным окнам (устойчиво для длинных сцен)

import re
import numpy as np
import pickle

# Единый стек модели и оконные утилиты
from embeddings import get_tok_mdl, tokenize_to_windows, encode_windows_batched  # [web:35][web:39][web:49][web:41]

# Инициализация общей модели/токенизатора
tok, mdl = get_tok_mdl()  # [web:35]
mdl.eval()  # [web:35]

# Категории и головы
CATS = ["violence","sexual","profanity","alcohol_drugs","scary"]  # [web:6]

with open("episode_heads.pkl","rb") as f:
    EP_HEADS = pickle.load(f)  # heads: {'cat': {'bin': LogReg, 'sev': Ridge}} [web:6]

# Параметры окон для эпизодов
WIN_LEN = 256      # короче, чем сценовые, чтобы точнее локализовать эпизод [web:39]
STRIDE  = 224      # перекрытие для непропускаемых фрагментов [web:39]
MAX_WINS = 16      # ограничение на количество окон для скорости [web:83]

def episode_windows_vecs(text: str):
    """
    Текст сцены -> токенные окна -> батчевый эмбеддинг окон (mean-pool).
    """
    ids, attn = tokenize_to_windows(text, max_len=WIN_LEN, stride=STRIDE)  # [Nw, L] [Nw, L] [web:39]
    # Ограничим число окон для стабильного времени и VRAM
    if ids.size(0) > MAX_WINS:
        ids = ids[:MAX_WINS]
        attn = attn[:MAX_WINS]
    V = encode_windows_batched(ids, attn, batch_size=8)  # [Nw, H] [web:35]
    return V  # [web:35]

def episode_aggregates_for_scene(scene_text: str):
    """
    Возвращает 30 признаков (5 категорий * 6 агрегатов): 
    [p_bin.max, p_bin.mean, top3mean(p_bin), p_sev.max, p_sev.mean, top3mean(p_sev)] по токенным окнам. 
    """
    V = episode_windows_vecs(scene_text)  # [Nw, H] [web:39]
    if V.ndim == 1:
        V = V[None, :]
    feats = []
    for cat in CATS:
        h = EP_HEADS[cat]
        # Бинарная вероятность «эпизод есть» по каждому окну
        p_bin = h["bin"].predict_proba(V)[:, 1]  # [Nw] [web:6]
        # Регрессия степени серьёзности 0..1 по каждому окну
        p_sev = np.clip(h["sev"].predict(V), 0.0, 1.0)  # [Nw] [web:6]
        # Агрегаты: max/mean и top-3 mean устойчиво отражают пик и общую «нагруженность»
        top3b = np.sort(p_bin)[-3:] if len(p_bin) >= 3 else p_bin  # [web:39]
        top3s = np.sort(p_sev)[-3:] if len(p_sev) >= 3 else p_sev  # [web:39]
        feats += [
            float(p_bin.max()),
            float(p_bin.mean()),
            float(np.mean(top3b)),
            float(p_sev.max()),
            float(p_sev.mean()),
            float(np.mean(top3s)),
        ]
    return np.array(feats, dtype=float)  # 5*6=30 фич [web:6]
