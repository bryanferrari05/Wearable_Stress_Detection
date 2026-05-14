import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# 1. Caricamento dati S2
# =========================

path = Path("S2") / "S2.pkl"

with open(path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

bvp = data["signal"]["wrist"]["BVP"].ravel()
labels = data["label"]

# Frequenze di campionamento
fs_bvp = 64      # BVP wrist = 64 Hz
fs_label = 700   # label = 700 Hz

# Parametri finestra
window_sec = 60
step_sec = 30

# Soglia di purezza della label
purity_threshold = 0.70

total_seconds = len(labels) // fs_label

rows = []

# =========================
# 2. Creazione finestre
# =========================

for start_sec in range(0, total_seconds - window_sec, step_sec):
    end_sec = start_sec + window_sec

    # Finestra BVP
    bvp_start = start_sec * fs_bvp
    bvp_end = end_sec * fs_bvp
    bvp_window = bvp[bvp_start:bvp_end]

    # Finestra label
    label_start = start_sec * fs_label
    label_end = end_sec * fs_label
    label_window = labels[label_start:label_end]

    # Se la finestra BVP o label è vuota, salto
    if len(bvp_window) == 0 or len(label_window) == 0:
        continue

    # Label prevalente nella finestra
    values, counts = np.unique(label_window, return_counts=True)

    majority_index = np.argmax(counts)
    majority_label = values[majority_index]

    # Percentuale della label prevalente nella finestra
    majority_ratio = counts[majority_index] / len(label_window)

    # Tengo solo:
    # 1 = baseline
    # 2 = stress
    # 3 = amusement
    if majority_label not in [1, 2, 3]:
        continue

    # Scarto finestre troppo miste
    # Tengo solo finestre dove almeno il 70% appartiene alla label prevalente
    if majority_ratio < purity_threshold:
        continue

    # Task binario:
    # stress = 1
    # non-stress = 0
    binary_label = 1 if majority_label == 2 else 0

    # =========================
    # 3. Feature extraction
    # =========================

    row = {
        "subject": data["subject"],
        "start_sec": start_sec,
        "end_sec": end_sec,
        "window_sec": window_sec,
        "step_sec": step_sec,

        "bvp_mean": np.mean(bvp_window),
        "bvp_std": np.std(bvp_window),
        "bvp_min": np.min(bvp_window),
        "bvp_max": np.max(bvp_window),
        "bvp_range": np.max(bvp_window) - np.min(bvp_window),
        "bvp_median": np.median(bvp_window),
        "bvp_energy": np.mean(bvp_window ** 2),

        "original_label": majority_label,
        "label": binary_label,
        "label_purity": majority_ratio
    }

    rows.append(row)

# =========================
# 4. Creazione DataFrame
# =========================

df = pd.DataFrame(rows)

print(df.head())
print("\nShape tabella:", df.shape)

print("\nDistribuzione label binarie:")
print(df["label"].value_counts())

print("\nDistribuzione label originali:")
print(df["original_label"].value_counts())

print("\nPurezza minima delle finestre:")
print(df["label_purity"].min())

# Salvataggio CSV
df.to_csv("bvp_features_s2.csv", index=False)

print("\nFile salvato: bvp_features_s2.csv")