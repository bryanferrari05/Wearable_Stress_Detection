import pickle
import numpy as np
from pathlib import Path

# Percorso del file S2.pkl
path = Path("S2") / "S2.pkl"

# Apro il file pickle
with open(path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Informazioni generali
print("TIPO DATA:", type(data))
print("CHIAVI PRINCIPALI:", data.keys())
print("SOGGETTO:", data["subject"])

# Segnali disponibili
print("\nDISPOSITIVI DISPONIBILI:")
print(data["signal"].keys())

print("\nSEGNALI CHEST:")
for name, values in data["signal"]["chest"].items():
    print(name, values.shape)

print("\nSEGNALI WRIST:")
for name, values in data["signal"]["wrist"].items():
    print(name, values.shape)

# Label
labels = data["label"]

print("\nLABEL:")
print("Shape:", labels.shape)
print("Valori presenti:", np.unique(labels, return_counts=True))

# Primo segnale che ci interessa: BVP wrist
bvp = data["signal"]["wrist"]["BVP"].ravel()

print("\nBVP WRIST:")
print("Shape:", bvp.shape)
print("Primi 10 valori:", bvp[:10])