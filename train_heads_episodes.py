# train_episode_head.py — устойчивое обучение эпизодных голов
import pandas as pd
import numpy as np
import pickle
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ===== Config =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "ai-forever/ruRoberta-large"
CATS = ["violence","sexual","profanity","alcohol_drugs","scary"]
EP_CSV = "episodes.csv"  # при необходимости поменяй путь

# ===== Embedding helpers =====
def embed_batch(texts, tok, mdl, max_len=512):
    mdl.eval()
    vecs = []
    with torch.no_grad():
        for t in texts:
            t = t if isinstance(t, str) else ""
            x = tok(t[:2000], return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
            hs = mdl(**x).last_hidden_state           # [1, T, H]
            attn = (x["attention_mask"].unsqueeze(-1) > 0).float()  # [1, T, 1]
            v = (hs * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)  # mean-pool
            vecs.append(v.cpu().numpy())
    return np.vstack(vecs)

def main(ep_csv=EP_CSV):
    # 1) Load data
    df = pd.read_csv(ep_csv)
    df["text"] = df["text"].fillna("")

    # 2) Embed episodes
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(DEVICE)
    print("Embedding episodes...")
    X = embed_batch(df["text"].tolist(), tok, mdl)  # shape: [N, H]

    heads = {}
    for cat in CATS:
        print(f"\n=== Category: {cat} ===")
        y_bin = df[cat].values.astype(int)
        y_sev = df[f"sev_{cat}_num"].values.astype(float)

        # Проверим распределение классов
        unique, counts = np.unique(y_bin, return_counts=True)
        class_ok = (len(unique) == 2) and (counts.min() >= 2)

        # Разделение
        if class_ok:
            Xtr, Xva, ytr, yva, ysev_tr, ysev_va = train_test_split(
                X, y_bin, y_sev, test_size=0.2, random_state=42, stratify=y_bin
            )
        else:
            # Без стратификации и с предупреждением
            Xtr, Xva, ytr, yva, ysev_tr, ysev_va = train_test_split(
                X, y_bin, y_sev, test_size=0.2, random_state=42
            )
            msg = f"Warning: unbalanced or single-class data for {cat}. unique={unique}, counts={counts}"
            print(msg)

        # 3) Бинарная голова
        clf_bin = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_bin.fit(Xtr, ytr)

        # 4) Регрессия по severity
        reg_sev = Ridge(alpha=1.0)
        reg_sev.fit(Xtr, ysev_tr)

        # 5) Отчет (аккуратный)
        try:
            uniq_val = np.unique(yva)
            if len(uniq_val) == 2:
                print(classification_report(yva, clf_bin.predict(Xva), target_names=["0","1"]))
            else:
                acc = float((clf_bin.predict(Xva) == yva).mean())
                print(f"Validation has single class {int(uniq_val[0])}; acc={acc:.3f}, pos_rate_train={ytr.mean():.3f}")
        except Exception as e:
            print(f"Report error for {cat}: {e}")

        heads[cat] = {"bin": clf_bin, "sev": reg_sev}

    # 6) Save
    with open("episode_heads.pkl","wb") as f:
        pickle.dump(heads, f)
    print("\n✅ Saved: episode_heads.pkl")

if __name__ == "__main__":
    main()
