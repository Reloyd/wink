import numpy as np

X = np.load("data_X.npy")
print("X:", X.shape)

for cat in ["violence","sexual","profanity","alcohol_drugs","scary"]:
    y = np.load(f"data_y_{cat}.npy")
    print(cat, y.shape)
