import re, numpy as np, torch, pickle
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "ai-forever/ruRoberta-large"
CATS = ["violence","sexual","profanity","alcohol_drugs","scary"]

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL).to(DEVICE)
mdl.eval()

with open("episode_heads.pkl","rb") as f:
    EP_HEADS = pickle.load(f)

SENT_SPLIT = re.compile(r'([.!?…]+[\"»”\')]*\s+)')

def embed_mean(text):
    x = tok(text[:2000], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        hs = mdl(**x).last_hidden_state
        attn = (x["attention_mask"].unsqueeze(-1) > 0).float()
        v = (hs * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
        return v.cpu().numpy().ravel()

def split_windows(text, left_right=1):
    parts = SENT_SPLIT.split(text)
    sents = []
    cur = ""
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        sep = parts[i+1] if i+1 < len(parts) else ""
        cur += chunk + (sep or "")
        if sep:
            sents.append(cur)
            cur = ""
    if cur.strip():
        sents.append(cur)
    # формируем окна по 3 предложения (предыдущее, текущее, следующее)
    wins = []
    for i in range(len(sents)):
        start = max(0, i - left_right)
        end = min(len(sents), i + 1 + left_right)
        wins.append("".join(sents[start:end]).strip())
    return wins[:12] if wins else [text]

def episode_aggregates_for_scene(scene_text):
    wins = split_windows(scene_text, left_right=1)
    vecs = [embed_mean(w) for w in wins]
    V = np.vstack(vecs)
    feats = []
    for cat in CATS:
        h = EP_HEADS[cat]
        p_bin = h["bin"].predict_proba(V)[:,1]
        p_sev = np.clip(h["sev"].predict(V), 0.0, 1.0)
        top3b = np.sort(p_bin)[-3:] if len(p_bin)>=3 else p_bin
        top3s = np.sort(p_sev)[-3:] if len(p_sev)>=3 else p_sev
        feats += [p_bin.max(), p_bin.mean(), float(np.mean(top3b)),
                  p_sev.max(), p_sev.mean(), float(np.mean(top3s))]
    return np.array(feats, dtype=float)  # 5 категорий * 6 = 30 фич
