# train_heads.py (замена цикла обучения)
import numpy as np, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train(prefix="data"):
    X = np.load(f"{prefix}_X.npy")
    cats = ["violence","sexual","profanity","alcohol_drugs","scary"]
    heads = {}
    for cat in cats:
        y = np.load(f"{prefix}_y_{cat}.npy")
        # если в метках один класс, чуть «разбавим» отчёт
        strat = y if (y.sum() > 0 and (len(y)-y.sum()) > 0) else None
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, ytr)

        # безопасный отчёт
        uniq = np.unique(yva)
        try:
            print(f"\n{cat}:")
            if len(uniq) == 2:
                print(classification_report(yva, clf.predict(Xva), target_names=["0","1"]))
            else:
                # один класс в валидации
                y_pred = clf.predict(Xva)
                acc = (y_pred == yva).mean()
                print(f"Валидация содержит один класс {int(uniq[0])}; accuracy={acc:.3f}, pos_rate_train={ytr.mean():.3f}")
        except Exception as e:
            print(f"{cat}: отчет пропущен ({e})")

        heads[cat] = clf

    with open("heads.pkl","wb") as f:
        pickle.dump(heads,f)
    print("OK: heads.pkl saved")

if __name__ == "__main__":
    train()
