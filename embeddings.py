# embeddings.py — общие утилиты для токенизации, оконного эмбеддинга и кэша

import os
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Кэш (можно поменять директорию при деплое)
try:
    from joblib import Memory
    CACHE_DIR = os.environ.get("SCRIPT_CACHE_DIR", "./.cache_script_rating")
    memory = Memory(CACHE_DIR, verbose=0)
except Exception:
    memory = None

# Модель (синглтон)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_NAME = "ai-forever/ruRoberta-large"

_tok = None
_mdl = None

def get_tok_mdl():
    global _tok, _mdl
    if _tok is None or _mdl is None:
        _tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _mdl = AutoModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
        _mdl.eval()
    return _tok, _mdl

# Хелпер: хэш текста + параметров окна (для кэша)
def _hash_text_and_params(text: str, max_len: int, stride: int) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8", errors="ignore"))
    h.update(f"|{max_len}|{stride}".encode())
    return h.hexdigest()

# Токенизация в окна по токенам
def tokenize_to_windows(text: str, max_len: int = 384, stride: int = 320):
    tok, _ = get_tok_mdl()
    enc = tok(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True
    )
    input_ids = enc["input_ids"][0]          # [T]
    attn = torch.ones_like(input_ids)        # потом нарежем под окна

    T = input_ids.size(0)
    if T <= max_len:
        return input_ids.unsqueeze(0), torch.ones(1, T, dtype=torch.long)  # [1, T], [1, T]

    # создаём окно с [CLS]...[SEP] в пределах max_len
    starts = list(range(0, max(1, T - max_len + 1), stride))
    if starts[-1] != T - max_len:
        starts.append(T - max_len)

    windows_ids = []
    windows_attn = []
    for s in starts:
        e = s + max_len
        ids_win = input_ids[s:e]
        attn_win = torch.ones_like(ids_win)
        windows_ids.append(ids_win)
        windows_attn.append(attn_win)
    input_ids_stacked = torch.nn.utils.rnn.pad_sequence(windows_ids, batch_first=True, padding_value=tok.pad_token_id)
    attn_stacked = torch.nn.utils.rnn.pad_sequence(windows_attn, batch_first=True, padding_value=0)
    return input_ids_stacked, attn_stacked   # [Nw, L], [Nw, L]

# Батчевый энкодинг окон
def encode_windows_batched(input_ids, attention_mask, batch_size: int = 8):
    tok, mdl = get_tok_mdl()
    Hs = []
    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            ids = input_ids[i:i+batch_size].to(_DEVICE)
            attn = attention_mask[i:i+batch_size].to(_DEVICE)
            out = mdl(input_ids=ids, attention_mask=attn)
            hs = out.last_hidden_state                      # [B, L, H]
            # mean-pool по валидным токенам
            mask = (attn.unsqueeze(-1) > 0).float()         # [B, L, 1]
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, H]
            Hs.append(pooled.detach().cpu())
    V = torch.cat(Hs, dim=0).cpu().numpy()                  # [Nw, H]
    return V

# Агрегация окон в вектор сцены
def aggregate_windows(V: np.ndarray, topk: int = 3):
    if V.ndim == 1:
        V = V[None, :]
    mean_vec = V.mean(axis=0)
    max_vec = V.max(axis=0)
    # top-k по L2 норме
    norms = np.linalg.norm(V, axis=1)
    k = min(topk, V.shape[0])
    top_idx = np.argsort(norms)[-k:]
    topk_mean = V[top_idx].mean(axis=0) if k > 0 else mean_vec
    return np.concatenate([mean_vec, max_vec, topk_mean], axis=0)  # 3H

# Полный пайплайн: текст → окна → эмбеддинги → агрегат
def scene_vector(text: str, max_len: int = 384, stride: int = 320, batch_size: int = 8, use_cache: bool = True):
    # Кэшируем на уровне эмбеддингов окон
    def _compute():
        ids, attn = tokenize_to_windows(text, max_len=max_len, stride=stride)
        V = encode_windows_batched(ids, attn, batch_size=batch_size)
        agg = aggregate_windows(V, topk=3)
        return agg

    if memory is not None and use_cache:
        key = _hash_text_and_params(text, max_len, stride)
        cached_fn = memory.cache(_compute)
        return cached_fn()
    else:
        return _compute()
