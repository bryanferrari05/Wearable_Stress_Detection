import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


DATA_RAW_DIR_CANDIDATES = (Path("data_raw"), Path("WESAD"), Path("."))
OUTPUT_DIR = Path("data_features")
OUTPUT_FILE = OUTPUT_DIR / "bvp_features_all_60s.csv"
TRAIN_OUTPUT_FILE = OUTPUT_DIR / "bvp_features_train_s2_s16_60s.csv"
TEST_OUTPUT_FILE = OUTPUT_DIR / "bvp_features_test_s17_60s.csv"

FS_BVP = 64
FS_LABEL = 700
WINDOW_SEC = 60
STEP_SEC = 30
PURITY_THRESHOLD = 0.70
MIN_HEART_RATE_BPM = 40
MAX_HEART_RATE_BPM = 180
PEAK_PROMINENCE_STD = 0.35

VALID_ORIGINAL_LABELS = [1, 2, 3]
LABEL_MAPPING = {
    1: 0,  # baseline -> non-stress
    2: 1,  # stress -> stress
    3: 0,  # amusement -> non-stress
}

EXCLUDED_SUBJECTS = {"S1", "S12"}
VALID_SUBJECTS = {
    f"S{subject_id}"
    for subject_id in range(2, 18)
    if f"S{subject_id}" not in EXCLUDED_SUBJECTS
}
TRAIN_SUBJECTS = VALID_SUBJECTS - {"S17"}
TEST_SUBJECTS = {"S17"}


def calculate_signal_features(values, prefix):
    signal = np.asarray(values, dtype=float)
    minimum = np.min(signal)
    maximum = np.max(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    centered = signal - mean

    if std == 0:
        skew = 0.0
        kurtosis = 0.0
    else:
        normalized = centered / std
        skew = np.mean(normalized**3)
        kurtosis = np.mean(normalized**4) - 3

    q25, q75 = np.percentile(signal, [25, 75])

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_range": maximum - minimum,
        f"{prefix}_median": np.median(signal),
        f"{prefix}_iqr": q75 - q25,
        f"{prefix}_rms": np.sqrt(np.mean(signal**2)),
        f"{prefix}_energy": np.sum(signal**2),
        f"{prefix}_skew": skew,
        f"{prefix}_kurtosis": kurtosis,
    }


def empty_bvp_peak_features():
    return {
        "bvp_peak_count": 0,
        "bvp_valid_ibi_count": 0,
        "bvp_peak_prominence_mean": 0.0,
        "bvp_peak_prominence_std": 0.0,
        "bvp_ibi_mean": 0.0,
        "bvp_ibi_std": 0.0,
        "bvp_ibi_min": 0.0,
        "bvp_ibi_max": 0.0,
        "bvp_hr_mean": 0.0,
        "bvp_hr_std": 0.0,
        "bvp_hr_min": 0.0,
        "bvp_hr_max": 0.0,
        "bvp_rmssd": 0.0,
        "bvp_sdnn": 0.0,
    }


def detect_bvp_peaks(normalized_signal):
    min_distance_samples = int(round(FS_BVP * 60 / MAX_HEART_RATE_BPM))
    return find_peaks(
        normalized_signal,
        distance=min_distance_samples,
        prominence=PEAK_PROMINENCE_STD,
    )


def get_peak_candidate(signal):
    signal_std = np.std(signal)

    if signal_std == 0:
        return np.array([], dtype=int), {"prominences": np.array([], dtype=float)}

    normalized = (signal - np.median(signal)) / signal_std
    positive_peaks, positive_props = detect_bvp_peaks(normalized)
    negative_peaks, negative_props = detect_bvp_peaks(-normalized)

    min_ibi_sec = 60 / MAX_HEART_RATE_BPM
    max_ibi_sec = 60 / MIN_HEART_RATE_BPM

    def candidate_score(peaks):
        ibi_seconds = np.diff(peaks) / FS_BVP
        valid_ibi = ibi_seconds[
            (ibi_seconds >= min_ibi_sec) & (ibi_seconds <= max_ibi_sec)
        ]
        return len(valid_ibi)

    positive_score = candidate_score(positive_peaks)
    negative_score = candidate_score(negative_peaks)

    if negative_score > positive_score:
        return negative_peaks, negative_props

    return positive_peaks, positive_props


def calculate_bvp_peak_features(values):
    signal = np.asarray(values, dtype=float)
    features = empty_bvp_peak_features()
    peaks, peak_props = get_peak_candidate(signal)

    features["bvp_peak_count"] = len(peaks)

    if len(peaks) < 3:
        return features

    ibi_seconds = np.diff(peaks) / FS_BVP
    min_ibi_sec = 60 / MAX_HEART_RATE_BPM
    max_ibi_sec = 60 / MIN_HEART_RATE_BPM
    valid_ibi_seconds = ibi_seconds[
        (ibi_seconds >= min_ibi_sec) & (ibi_seconds <= max_ibi_sec)
    ]

    if len(valid_ibi_seconds) < 2:
        return features

    instant_hr = 60 / valid_ibi_seconds
    ibi_diff = np.diff(valid_ibi_seconds)
    prominences = peak_props.get("prominences", np.array([], dtype=float))

    features.update(
        {
            "bvp_valid_ibi_count": len(valid_ibi_seconds),
            "bvp_peak_prominence_mean": np.mean(prominences)
            if len(prominences) > 0
            else 0.0,
            "bvp_peak_prominence_std": np.std(prominences)
            if len(prominences) > 0
            else 0.0,
            "bvp_ibi_mean": np.mean(valid_ibi_seconds),
            "bvp_ibi_std": np.std(valid_ibi_seconds),
            "bvp_ibi_min": np.min(valid_ibi_seconds),
            "bvp_ibi_max": np.max(valid_ibi_seconds),
            "bvp_hr_mean": np.mean(instant_hr),
            "bvp_hr_std": np.std(instant_hr),
            "bvp_hr_min": np.min(instant_hr),
            "bvp_hr_max": np.max(instant_hr),
            "bvp_rmssd": np.sqrt(np.mean(ibi_diff**2)) if len(ibi_diff) > 0 else 0.0,
            "bvp_sdnn": np.std(valid_ibi_seconds),
        }
    )

    return features


def get_visible_deprecation_warning():
    numpy_exceptions = getattr(np, "exceptions", np)
    return getattr(numpy_exceptions, "VisibleDeprecationWarning", Warning)


def subject_sort_key(subject_folder):
    return int(subject_folder.name[1:])


def is_valid_subject_folder(folder):
    return folder.is_dir() and folder.name in VALID_SUBJECTS


def load_subject_pickle(subject_folder):
    subject_id = subject_folder.name
    pickle_path = subject_folder / f"{subject_id}.pkl"

    if not pickle_path.exists():
        print(f"Warning: salto {subject_id}, file non trovato: {pickle_path}")
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=get_visible_deprecation_warning()
            )
            with open(pickle_path, "rb") as file:
                return pickle.load(file, encoding="latin1")
    except Exception as error:
        print(f"Warning: impossibile caricare {pickle_path}: {error}")
        return None


def find_subject_folders():
    for data_raw_dir in DATA_RAW_DIR_CANDIDATES:
        if not data_raw_dir.exists():
            continue

        subject_folders = sorted(
            (
                folder
                for folder in data_raw_dir.iterdir()
                if is_valid_subject_folder(folder)
            ),
            key=subject_sort_key,
        )

        if subject_folders:
            if data_raw_dir != DATA_RAW_DIR_CANDIDATES[0]:
                print(f"Nota: uso {data_raw_dir} come cartella dati.")
            return subject_folders

    searched_dirs = "\n".join(
        f"- {data_raw_dir.resolve()}" for data_raw_dir in DATA_RAW_DIR_CANDIDATES
    )
    raise FileNotFoundError(
        "Nessuna cartella soggetto valida trovata.\n"
        f"Cartelle controllate:\n{searched_dirs}\n"
        "Inserisci le cartelle WESAD in data_raw/, ad esempio data_raw/S2/S2.pkl. "
        "I soggetti S1 e S12 sono esclusi."
    )


def create_bvp_features_for_subject(
    subject_folder,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
    purity_threshold=PURITY_THRESHOLD,
):
    data = load_subject_pickle(subject_folder)

    if data is None:
        return pd.DataFrame()

    try:
        bvp = np.asarray(data["signal"]["wrist"]["BVP"]).ravel()
        labels = np.asarray(data["label"]).ravel()
        subject = str(data.get("subject", subject_folder.name))
    except KeyError as error:
        subject = str(data.get("subject", subject_folder.name))
        print(f"Warning: salto {subject}, campo mancante: {error}")
        return pd.DataFrame()

    if len(bvp) == 0 or len(labels) == 0:
        print(f"Warning: salto {subject}, BVP o label vuoti.")
        return pd.DataFrame()

    duration_sec = min(len(bvp) / FS_BVP, len(labels) / FS_LABEL)
    max_start_sec = int(np.floor(duration_sec - window_sec))

    if max_start_sec < 0:
        print(f"Warning: salto {subject}, durata inferiore a {window_sec} secondi.")
        return pd.DataFrame()

    rows = []

    for start_sec in range(0, max_start_sec + 1, step_sec):
        end_sec = start_sec + window_sec

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

        if label_purity < purity_threshold:
            continue

        bvp_features = calculate_signal_features(bvp_window, "bvp")
        bvp_features.update(calculate_bvp_peak_features(bvp_window))

        rows.append(
            {
                "subject": subject,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "window_sec": window_sec,
                "step_sec": step_sec,
                **bvp_features,
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
    subject_folders = find_subject_folders()

    subject_dataframes = []

    for subject_folder in subject_folders:
        subject_df = create_bvp_features_for_subject(subject_folder)

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
