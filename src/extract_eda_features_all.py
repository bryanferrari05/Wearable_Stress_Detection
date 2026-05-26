import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, savgol_filter, sosfiltfilt


DATA_RAW_DIR_CANDIDATES = (Path("data_raw"), Path("WESAD"), Path("."))
OUTPUT_DIR = Path("data_features")
OUTPUT_FILE = OUTPUT_DIR / "eda_features_all_60s.csv"
TRAIN_OUTPUT_FILE = OUTPUT_DIR / "eda_features_train_s2_s16_60s.csv"
TEST_OUTPUT_FILE = OUTPUT_DIR / "eda_features_test_s17_60s.csv"

FS_EDA = 4
FS_LABEL = 700
WINDOW_SEC = 60
STEP_SEC = 30
PURITY_THRESHOLD = 0.70

LOWPASS_CUTOFF_HZ = 1.0
TONIC_WINDOW_SEC = 10
SCR_MIN_DISTANCE_SEC = 1
SCR_MIN_AMPLITUDE = 0.01

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


def calculate_slope(values):
    signal = np.asarray(values, dtype=float)
    if len(signal) < 2:
        return 0.0
    time_seconds = np.arange(len(signal), dtype=float) / FS_EDA
    return float(np.polyfit(time_seconds, signal, deg=1)[0])


def filter_eda_signal(values):
    signal = np.asarray(values, dtype=float)
    sos = butter(2, LOWPASS_CUTOFF_HZ, btype="lowpass", fs=FS_EDA, output="sos")
    return sosfiltfilt(sos, signal)


def extract_tonic_phasic(filtered_signal):
    desired_window = int(round(TONIC_WINDOW_SEC * FS_EDA))
    if desired_window % 2 == 0:
        desired_window += 1
    max_odd_window = len(filtered_signal) if len(filtered_signal) % 2 else len(filtered_signal) - 1
    window_length = min(desired_window, max_odd_window)

    if window_length < 5:
        tonic = np.full_like(filtered_signal, np.mean(filtered_signal))
    else:
        tonic = savgol_filter(filtered_signal, window_length, polyorder=2)

    phasic = filtered_signal - tonic
    return tonic, phasic


def empty_scr_features():
    return {
        "eda_scr_count": 0,
        "eda_scr_rate_per_min": 0.0,
        "eda_scr_amplitude_mean": 0.0,
        "eda_scr_amplitude_std": 0.0,
        "eda_scr_amplitude_max": 0.0,
        "eda_scr_prominence_mean": 0.0,
    }


def calculate_eda_features(values):
    signal = np.asarray(values, dtype=float)
    filtered = filter_eda_signal(signal)
    tonic, phasic = extract_tonic_phasic(filtered)
    positive_phasic = np.maximum(phasic, 0.0)
    derivative = np.diff(filtered) * FS_EDA

    features = calculate_signal_features(signal, "eda")
    features.update(
        {
            "eda_slope": calculate_slope(filtered),
            "eda_derivative_mean": np.mean(derivative) if len(derivative) else 0.0,
            "eda_derivative_std": np.std(derivative) if len(derivative) else 0.0,
            "eda_derivative_max": np.max(derivative) if len(derivative) else 0.0,
            "eda_tonic_mean": np.mean(tonic),
            "eda_tonic_std": np.std(tonic),
            "eda_tonic_slope": calculate_slope(tonic),
            "eda_phasic_mean": np.mean(positive_phasic),
            "eda_phasic_std": np.std(positive_phasic),
            "eda_phasic_max": np.max(positive_phasic),
            "eda_phasic_auc": np.sum(positive_phasic) / FS_EDA,
        }
    )

    peaks, properties = find_peaks(
        positive_phasic,
        height=SCR_MIN_AMPLITUDE,
        prominence=SCR_MIN_AMPLITUDE,
        distance=int(round(SCR_MIN_DISTANCE_SEC * FS_EDA)),
    )
    scr_features = empty_scr_features()

    if len(peaks):
        amplitudes = properties["peak_heights"]
        prominences = properties["prominences"]
        window_duration_sec = len(signal) / FS_EDA
        scr_features.update(
            {
                "eda_scr_count": len(peaks),
                "eda_scr_rate_per_min": len(peaks) * 60 / window_duration_sec,
                "eda_scr_amplitude_mean": np.mean(amplitudes),
                "eda_scr_amplitude_std": np.std(amplitudes),
                "eda_scr_amplitude_max": np.max(amplitudes),
                "eda_scr_prominence_mean": np.mean(prominences),
            }
        )

    features.update(scr_features)
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


def create_eda_features_for_subject(
    subject_folder,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
    purity_threshold=PURITY_THRESHOLD,
):
    data = load_subject_pickle(subject_folder)

    if data is None:
        return pd.DataFrame()

    try:
        eda = np.asarray(data["signal"]["wrist"]["EDA"]).ravel()
        labels = np.asarray(data["label"]).ravel()
        subject = str(data.get("subject", subject_folder.name))
    except KeyError as error:
        subject = str(data.get("subject", subject_folder.name))
        print(f"Warning: salto {subject}, campo mancante: {error}")
        return pd.DataFrame()

    if len(eda) == 0 or len(labels) == 0:
        print(f"Warning: salto {subject}, EDA o label vuoti.")
        return pd.DataFrame()

    duration_sec = min(len(eda) / FS_EDA, len(labels) / FS_LABEL)
    max_start_sec = int(np.floor(duration_sec - window_sec))

    if max_start_sec < 0:
        print(f"Warning: salto {subject}, durata inferiore a {window_sec} secondi.")
        return pd.DataFrame()

    rows = []

    for start_sec in range(0, max_start_sec + 1, step_sec):
        end_sec = start_sec + window_sec

        eda_start = int(round(start_sec * FS_EDA))
        eda_end = int(round(end_sec * FS_EDA))
        label_start = int(round(start_sec * FS_LABEL))
        label_end = int(round(end_sec * FS_LABEL))

        eda_window = eda[eda_start:eda_end]
        label_window = labels[label_start:label_end]

        if len(eda_window) == 0 or len(label_window) == 0:
            continue

        values, counts = np.unique(label_window, return_counts=True)
        majority_index = int(np.argmax(counts))
        original_label = int(values[majority_index])
        label_purity = counts[majority_index] / len(label_window)

        if original_label not in VALID_ORIGINAL_LABELS:
            continue

        if label_purity < purity_threshold:
            continue

        rows.append(
            {
                "subject": subject,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "window_sec": window_sec,
                "step_sec": step_sec,
                **calculate_eda_features(eda_window),
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
    print(f"Shape train: {train_df.shape}")
    print(f"Shape test: {test_df.shape}")

    if missing_train_subjects:
        print(f"Soggetti train attesi ma non presenti: {missing_train_subjects}")


def main():
    subject_folders = find_subject_folders()
    subject_dataframes = []

    for subject_folder in subject_folders:
        subject_df = create_eda_features_for_subject(subject_folder)

        if subject_df.empty:
            print(f"Warning: nessuna finestra valida per {subject_folder.name}.")
            continue

        print(f"{subject_folder.name}: {len(subject_df)} finestre valide")
        subject_dataframes.append(subject_df)

    if not subject_dataframes:
        raise ValueError(
            "Nessuna riga generata. Controlla che i file .pkl contengano EDA, "
            "label e finestre con original_label in [1, 2, 3] e purezza >= 0.70."
        )

    df = pd.concat(subject_dataframes, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFile salvato: {OUTPUT_FILE}")

    save_subject_split(df)

    print(f"\nShape DataFrame: {df.shape}")
    print(f"Numero soggetti: {df['subject'].nunique()}")
    print("\nDistribuzione label binarie:")
    print(df["label"].value_counts().sort_index())
    print(f"\nPurezza minima: {df['label_purity'].min()}")
    print(f"NaN totali: {int(df.isna().sum().sum())}")
    print("\nDistribuzione SCR count:")
    print(df["eda_scr_count"].describe())


if __name__ == "__main__":
    main()
