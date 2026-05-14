import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DATA_RAW_DIR = Path("data_raw")
FALLBACK_DATA_RAW_DIR = Path("WESAD")
OUTPUT_DIR = Path("data_features")
OUTPUT_FILE = OUTPUT_DIR / "bvp_features_all_60s.csv"
TRAIN_OUTPUT_FILE = OUTPUT_DIR / "bvp_features_train_s2_s16_60s.csv"
TEST_OUTPUT_FILE = OUTPUT_DIR / "bvp_features_test_s17_60s.csv"

FS_BVP = 64
FS_LABEL = 700
WINDOW_SEC = 60
STEP_SEC = 30
PURITY_THRESHOLD = 0.70

VALID_ORIGINAL_LABELS = [1, 2, 3]
LABEL_MAPPING = {
    1: 0,  # baseline -> non-stress
    2: 1,  # stress -> stress
    3: 0,  # amusement -> non-stress
}

TRAIN_SUBJECTS = {f"S{subject_id}" for subject_id in range(2, 17)}
TEST_SUBJECTS = {"S17"}


def load_subject_pickle(subject_folder):
    subject_id = subject_folder.name
    pickle_path = subject_folder / f"{subject_id}.pkl"

    if not pickle_path.exists():
        print(f"Warning: salto {subject_id}, file non trovato: {pickle_path}")
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=np.exceptions.VisibleDeprecationWarning
            )
            with open(pickle_path, "rb") as file:
                return pickle.load(file, encoding="latin1")
    except Exception as error:
        print(f"Warning: impossibile caricare {pickle_path}: {error}")
        return None


def get_data_raw_dir():
    if DATA_RAW_DIR.exists():
        return DATA_RAW_DIR

    if FALLBACK_DATA_RAW_DIR.exists():
        print(
            f"Nota: {DATA_RAW_DIR} non esiste. Uso {FALLBACK_DATA_RAW_DIR} "
            "come cartella dati."
        )
        return FALLBACK_DATA_RAW_DIR

    raise FileNotFoundError(
        f"Cartella dati non trovata: {DATA_RAW_DIR.resolve()}\n"
        f"Cartella alternativa non trovata: {FALLBACK_DATA_RAW_DIR.resolve()}\n"
        "Crea data_raw/ e inserisci le cartelle dei soggetti, ad esempio "
        "data_raw/S2/S2.pkl."
    )


def create_bvp_features_for_subject(data):
    try:
        bvp = np.asarray(data["signal"]["wrist"]["BVP"]).ravel()
        labels = np.asarray(data["label"]).ravel()
        subject = data["subject"]
    except KeyError as error:
        subject = data.get("subject", "soggetto_sconosciuto")
        print(f"Warning: salto {subject}, campo mancante: {error}")
        return pd.DataFrame()

    if len(bvp) == 0 or len(labels) == 0:
        print(f"Warning: salto {subject}, BVP o label vuoti.")
        return pd.DataFrame()

    duration_sec = min(len(bvp) / FS_BVP, len(labels) / FS_LABEL)
    max_start_sec = int(np.floor(duration_sec - WINDOW_SEC))

    if max_start_sec < 0:
        print(f"Warning: salto {subject}, durata inferiore a {WINDOW_SEC} secondi.")
        return pd.DataFrame()

    rows = []

    for start_sec in range(0, max_start_sec + 1, STEP_SEC):
        end_sec = start_sec + WINDOW_SEC

        bvp_start = int(round(start_sec * FS_BVP))
        bvp_end = int(round(end_sec * FS_BVP))
        label_start = int(round(start_sec * FS_LABEL))
        label_end = int(round(end_sec * FS_LABEL))

        bvp_window = bvp[bvp_start:bvp_end]
        label_window = labels[label_start:label_end]

        if len(bvp_window) == 0 or len(label_window) == 0:
            continue

        values, counts = np.unique(label_window, return_counts=True)
        majority_index = int(np.argmax(counts))
        original_label = int(values[majority_index])
        label_purity = counts[majority_index] / len(label_window)

        if original_label not in VALID_ORIGINAL_LABELS:
            continue

        if label_purity < PURITY_THRESHOLD:
            continue

        bvp_min = np.min(bvp_window)
        bvp_max = np.max(bvp_window)

        rows.append(
            {
                "subject": subject,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "window_sec": WINDOW_SEC,
                "step_sec": STEP_SEC,
                "bvp_mean": np.mean(bvp_window),
                "bvp_std": np.std(bvp_window),
                "bvp_min": bvp_min,
                "bvp_max": bvp_max,
                "bvp_range": bvp_max - bvp_min,
                "original_label": original_label,
                "label": LABEL_MAPPING[original_label],
                "label_purity": label_purity,
            }
        )

    return pd.DataFrame(rows)


def save_subject_split(df):
    train_df = df[df["subject"].isin(TRAIN_SUBJECTS)].copy()
    test_df = df[df["subject"].isin(TEST_SUBJECTS)].copy()
    missing_train_subjects = sorted(TRAIN_SUBJECTS - set(train_df["subject"].unique()))

    if train_df.empty:
        raise ValueError("Training set vuoto: nessun soggetto S2-S16 trovato.")

    if test_df.empty:
        raise ValueError("Test set vuoto: soggetto S17 non trovato.")

    train_df.to_csv(TRAIN_OUTPUT_FILE, index=False)
    test_df.to_csv(TEST_OUTPUT_FILE, index=False)

    print(f"\nFile train salvato: {TRAIN_OUTPUT_FILE}")
    print(f"File test salvato: {TEST_OUTPUT_FILE}")

    print("\nShape train:")
    print(train_df.shape)

    print("\nShape test:")
    print(test_df.shape)

    print("\nSoggetti train:")
    print(sorted(train_df["subject"].unique()))

    if missing_train_subjects:
        print("\nSoggetti train attesi ma non presenti nei dati:")
        print(missing_train_subjects)

    print("\nSoggetti test:")
    print(sorted(test_df["subject"].unique()))

    print("\nDistribuzione label train:")
    print(train_df["label"].value_counts().sort_index())

    print("\nDistribuzione label test:")
    print(test_df["label"].value_counts().sort_index())


def main():
    data_raw_dir = get_data_raw_dir()

    subject_folders = sorted(
        folder for folder in data_raw_dir.iterdir() if folder.is_dir()
    )

    if not subject_folders:
        raise ValueError(
            f"Nessuna cartella soggetto trovata dentro {data_raw_dir.resolve()}."
        )

    subject_dataframes = []

    for subject_folder in subject_folders:
        data = load_subject_pickle(subject_folder)

        if data is None:
            continue

        subject_df = create_bvp_features_for_subject(data)

        if subject_df.empty:
            print(f"Warning: nessuna finestra valida per {subject_folder.name}.")
            continue

        print(f"{subject_folder.name}: {len(subject_df)} finestre valide")
        subject_dataframes.append(subject_df)

    if not subject_dataframes:
        raise ValueError(
            "Nessuna riga generata. Controlla che i file .pkl contengano BVP, "
            "label e finestre con original_label in [1, 2, 3] e purezza >= 0.70."
        )

    df = pd.concat(subject_dataframes, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFile salvato: {OUTPUT_FILE}")

    save_subject_split(df)

    print("\nShape DataFrame:")
    print(df.shape)

    print("\nNumero soggetti:")
    print(df["subject"].nunique())

    print("\nDistribuzione label binarie:")
    print(df["label"].value_counts().sort_index())

    print("\nDistribuzione original_label:")
    print(df["original_label"].value_counts().sort_index())

    print("\nPurezza minima:")
    print(df["label_purity"].min())

    print("\nNumero di NaN per colonna:")
    print(df.isna().sum())

    print("\nDistribuzione per subject e label:")
    print(pd.crosstab(df["subject"], df["label"]))


if __name__ == "__main__":
    main()
