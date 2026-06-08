import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


DATA_RAW_DIR_CANDIDATES = (
    Path("data_raw"),
    Path("WESAD"),
    Path.home() / "Downloads" / "WESAD" / "WESAD",
    Path.home() / "Downloads" / "WESAD",
    Path("."),
)
OUTPUT_DIR = Path("data_features")
OUTPUT_FILE = OUTPUT_DIR / "resp_chest_features_all_60s.csv"
TRAIN_OUTPUT_FILE = OUTPUT_DIR / "resp_chest_features_train_s2_s16_60s.csv"
TEST_OUTPUT_FILE = OUTPUT_DIR / "resp_chest_features_test_s17_60s.csv"

FS_RESP = 700
FS_LABEL = 700
WINDOW_SEC = 60
STEP_SEC = 30
PURITY_THRESHOLD = 0.70

LOWCUT_HZ = 0.05
HIGHCUT_HZ = 2.0
WELCH_NPERSEG = 4096
MIN_BREATH_RATE_BPM = 3
MAX_BREATH_RATE_BPM = 45
RESP_PEAK_PROMINENCE_STD = 0.30
EPSILON = 1e-12

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

RESP_FEATURE_COLUMNS = [
    "resp_mean",
    "resp_std",
    "resp_range",
    "resp_median",
    "resp_iqr",
    "resp_skew",
    "resp_kurtosis",
    "resp_derivative_std",
    "resp_derivative_max_abs",
    "resp_second_derivative_std",
    "resp_peak_rate_per_min",
    "resp_trough_rate_per_min",
    "resp_peak_prominence_std",
    "resp_trough_prominence_std",
    "resp_breath_rate_mean",
    "resp_breath_rate_std",
    "resp_breath_rate_range",
    "resp_interval_rmssd",
    "resp_interval_sdnn",
    "resp_interval_cv",
    "resp_cycle_amplitude_mean",
    "resp_cycle_amplitude_std",
    "resp_dominant_rate_bpm",
    "resp_mean_frequency_hz",
    "resp_bandpower_0_15_0_40",
    "resp_bandpower_0_40_0_75",
    "resp_bandpower_0_75_2_00",
    "resp_relative_power_0_75_2_00",
]


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


def filter_resp_signal(values):
    signal = np.asarray(values, dtype=float)
    sos = butter(
        3,
        [LOWCUT_HZ, HIGHCUT_HZ],
        btype="bandpass",
        fs=FS_RESP,
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def calculate_derivative_features(filtered_signal):
    derivative = np.diff(filtered_signal) * FS_RESP
    second_derivative = np.diff(derivative) * FS_RESP if len(derivative) else np.array([])

    return {
        "resp_derivative_mean": np.mean(derivative) if len(derivative) else 0.0,
        "resp_derivative_std": np.std(derivative) if len(derivative) else 0.0,
        "resp_derivative_max_abs": (
            np.max(np.abs(derivative)) if len(derivative) else 0.0
        ),
        "resp_second_derivative_std": (
            np.std(second_derivative) if len(second_derivative) else 0.0
        ),
    }


def empty_breath_cycle_features():
    return {
        "resp_peak_count": 0,
        "resp_trough_count": 0,
        "resp_valid_interval_count": 0,
        "resp_peak_rate_per_min": 0.0,
        "resp_trough_rate_per_min": 0.0,
        "resp_peak_prominence_mean": 0.0,
        "resp_peak_prominence_std": 0.0,
        "resp_trough_prominence_mean": 0.0,
        "resp_trough_prominence_std": 0.0,
        "resp_breath_interval_mean": 0.0,
        "resp_breath_interval_std": 0.0,
        "resp_breath_interval_min": 0.0,
        "resp_breath_interval_max": 0.0,
        "resp_breath_rate_mean": 0.0,
        "resp_breath_rate_std": 0.0,
        "resp_breath_rate_min": 0.0,
        "resp_breath_rate_max": 0.0,
        "resp_breath_rate_range": 0.0,
        "resp_interval_rmssd": 0.0,
        "resp_interval_sdnn": 0.0,
        "resp_interval_cv": 0.0,
        "resp_cycle_amplitude_mean": 0.0,
        "resp_cycle_amplitude_std": 0.0,
        "resp_cycle_amplitude_min": 0.0,
        "resp_cycle_amplitude_max": 0.0,
    }


def detect_extrema(normalized_signal):
    min_distance_samples = int(round(FS_RESP * 60 / MAX_BREATH_RATE_BPM))
    peaks, peak_props = find_peaks(
        normalized_signal,
        distance=min_distance_samples,
        prominence=RESP_PEAK_PROMINENCE_STD,
    )
    troughs, trough_props = find_peaks(
        -normalized_signal,
        distance=min_distance_samples,
        prominence=RESP_PEAK_PROMINENCE_STD,
    )
    return peaks, peak_props, troughs, trough_props


def calculate_cycle_amplitudes(filtered_signal, peaks, troughs):
    if len(peaks) == 0 or len(troughs) == 0:
        return np.array([], dtype=float)

    amplitudes = []
    troughs = np.asarray(troughs)

    for peak in peaks:
        previous_troughs = troughs[troughs < peak]
        if len(previous_troughs) == 0:
            continue
        trough = previous_troughs[-1]
        amplitudes.append(filtered_signal[peak] - filtered_signal[trough])

    return np.asarray(amplitudes, dtype=float)


def calculate_breath_cycle_features(filtered_signal):
    features = empty_breath_cycle_features()
    signal_std = np.std(filtered_signal)
    window_duration_sec = len(filtered_signal) / FS_RESP

    if signal_std == 0 or window_duration_sec == 0:
        return features

    normalized = (filtered_signal - np.median(filtered_signal)) / signal_std
    peaks, peak_props, troughs, trough_props = detect_extrema(normalized)

    features["resp_peak_count"] = len(peaks)
    features["resp_trough_count"] = len(troughs)
    features["resp_peak_rate_per_min"] = len(peaks) * 60 / window_duration_sec
    features["resp_trough_rate_per_min"] = len(troughs) * 60 / window_duration_sec

    peak_prominences = peak_props.get("prominences", np.array([], dtype=float))
    trough_prominences = trough_props.get("prominences", np.array([], dtype=float))
    if len(peak_prominences):
        features["resp_peak_prominence_mean"] = np.mean(peak_prominences)
        features["resp_peak_prominence_std"] = np.std(peak_prominences)
    if len(trough_prominences):
        features["resp_trough_prominence_mean"] = np.mean(trough_prominences)
        features["resp_trough_prominence_std"] = np.std(trough_prominences)

    amplitudes = calculate_cycle_amplitudes(filtered_signal, peaks, troughs)
    if len(amplitudes):
        features["resp_cycle_amplitude_mean"] = np.mean(amplitudes)
        features["resp_cycle_amplitude_std"] = np.std(amplitudes)
        features["resp_cycle_amplitude_min"] = np.min(amplitudes)
        features["resp_cycle_amplitude_max"] = np.max(amplitudes)

    if len(peaks) < 3:
        return features

    intervals_sec = np.diff(peaks) / FS_RESP
    min_interval_sec = 60 / MAX_BREATH_RATE_BPM
    max_interval_sec = 60 / MIN_BREATH_RATE_BPM
    valid_intervals_sec = intervals_sec[
        (intervals_sec >= min_interval_sec) & (intervals_sec <= max_interval_sec)
    ]

    if len(valid_intervals_sec) < 2:
        return features

    breath_rates = 60 / valid_intervals_sec
    interval_diff = np.diff(valid_intervals_sec)
    interval_mean = np.mean(valid_intervals_sec)
    interval_sdnn = np.std(valid_intervals_sec)

    features.update(
        {
            "resp_valid_interval_count": len(valid_intervals_sec),
            "resp_breath_interval_mean": interval_mean,
            "resp_breath_interval_std": interval_sdnn,
            "resp_breath_interval_min": np.min(valid_intervals_sec),
            "resp_breath_interval_max": np.max(valid_intervals_sec),
            "resp_breath_rate_mean": np.mean(breath_rates),
            "resp_breath_rate_std": np.std(breath_rates),
            "resp_breath_rate_min": np.min(breath_rates),
            "resp_breath_rate_max": np.max(breath_rates),
            "resp_breath_rate_range": np.max(breath_rates) - np.min(breath_rates),
            "resp_interval_rmssd": (
                np.sqrt(np.mean(interval_diff**2)) if len(interval_diff) else 0.0
            ),
            "resp_interval_sdnn": interval_sdnn,
            "resp_interval_cv": interval_sdnn / interval_mean if interval_mean else 0.0,
        }
    )
    return features


def calculate_band_power(frequencies, power_spectrum, low_hz, high_hz):
    mask = (frequencies >= low_hz) & (frequencies < high_hz)

    if not np.any(mask):
        return 0.0

    return float(np.trapezoid(power_spectrum[mask], frequencies[mask]))


def empty_frequency_features():
    return {
        "resp_total_power": 0.0,
        "resp_dominant_frequency_hz": 0.0,
        "resp_dominant_rate_bpm": 0.0,
        "resp_mean_frequency_hz": 0.0,
        "resp_bandpower_0_05_0_15": 0.0,
        "resp_bandpower_0_15_0_40": 0.0,
        "resp_bandpower_0_40_0_75": 0.0,
        "resp_bandpower_0_75_2_00": 0.0,
        "resp_relative_power_0_05_0_15": 0.0,
        "resp_relative_power_0_15_0_40": 0.0,
        "resp_relative_power_0_40_0_75": 0.0,
        "resp_relative_power_0_75_2_00": 0.0,
    }


def calculate_frequency_domain_features(filtered_signal):
    nperseg = min(WELCH_NPERSEG, len(filtered_signal))
    frequencies, power_spectrum = welch(
        filtered_signal,
        fs=FS_RESP,
        nperseg=nperseg,
    )
    frequency_mask = (frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ)
    resp_frequencies = frequencies[frequency_mask]
    resp_power_spectrum = power_spectrum[frequency_mask]

    if len(resp_frequencies) == 0:
        return empty_frequency_features()

    total_power = float(np.trapezoid(resp_power_spectrum, resp_frequencies))
    power_sum = np.sum(resp_power_spectrum)
    weighted_power_sum = np.sum(resp_frequencies * resp_power_spectrum)

    if power_sum > 0:
        dominant_frequency = resp_frequencies[np.argmax(resp_power_spectrum)]
        mean_frequency = weighted_power_sum / power_sum
    else:
        dominant_frequency = 0.0
        mean_frequency = 0.0

    bandpower_0_05_0_15 = calculate_band_power(
        resp_frequencies, resp_power_spectrum, 0.05, 0.15
    )
    bandpower_0_15_0_40 = calculate_band_power(
        resp_frequencies, resp_power_spectrum, 0.15, 0.40
    )
    bandpower_0_40_0_75 = calculate_band_power(
        resp_frequencies, resp_power_spectrum, 0.40, 0.75
    )
    bandpower_0_75_2_00 = calculate_band_power(
        resp_frequencies, resp_power_spectrum, 0.75, 2.00
    )

    return {
        "resp_total_power": total_power,
        "resp_dominant_frequency_hz": dominant_frequency,
        "resp_dominant_rate_bpm": dominant_frequency * 60,
        "resp_mean_frequency_hz": mean_frequency,
        "resp_bandpower_0_05_0_15": bandpower_0_05_0_15,
        "resp_bandpower_0_15_0_40": bandpower_0_15_0_40,
        "resp_bandpower_0_40_0_75": bandpower_0_40_0_75,
        "resp_bandpower_0_75_2_00": bandpower_0_75_2_00,
        "resp_relative_power_0_05_0_15": (
            bandpower_0_05_0_15 / total_power if total_power > EPSILON else 0.0
        ),
        "resp_relative_power_0_15_0_40": (
            bandpower_0_15_0_40 / total_power if total_power > EPSILON else 0.0
        ),
        "resp_relative_power_0_40_0_75": (
            bandpower_0_40_0_75 / total_power if total_power > EPSILON else 0.0
        ),
        "resp_relative_power_0_75_2_00": (
            bandpower_0_75_2_00 / total_power if total_power > EPSILON else 0.0
        ),
    }


def calculate_resp_features(values):
    signal = np.asarray(values, dtype=float)
    filtered = filter_resp_signal(signal)

    features = calculate_signal_features(filtered, "resp")
    features.update(calculate_derivative_features(filtered))
    features.update(calculate_breath_cycle_features(filtered))
    features.update(calculate_frequency_domain_features(filtered))
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


def create_resp_features_for_subject(
    subject_folder,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
    purity_threshold=PURITY_THRESHOLD,
):
    data = load_subject_pickle(subject_folder)

    if data is None:
        return pd.DataFrame()

    try:
        resp = np.asarray(data["signal"]["chest"]["Resp"]).ravel()
        labels = np.asarray(data["label"]).ravel()
        subject = str(data.get("subject", subject_folder.name))
    except KeyError as error:
        subject = str(data.get("subject", subject_folder.name))
        print(f"Warning: salto {subject}, campo mancante: {error}")
        return pd.DataFrame()

    if len(resp) == 0 or len(labels) == 0:
        print(f"Warning: salto {subject}, RESP o label vuoti.")
        return pd.DataFrame()

    duration_sec = min(len(resp) / FS_RESP, len(labels) / FS_LABEL)
    max_start_sec = int(np.floor(duration_sec - window_sec))

    if max_start_sec < 0:
        print(f"Warning: salto {subject}, durata inferiore a {window_sec} secondi.")
        return pd.DataFrame()

    rows = []

    for start_sec in range(0, max_start_sec + 1, step_sec):
        end_sec = start_sec + window_sec

        resp_start = int(round(start_sec * FS_RESP))
        resp_end = int(round(end_sec * FS_RESP))
        label_start = int(round(start_sec * FS_LABEL))
        label_end = int(round(end_sec * FS_LABEL))

        resp_window = resp[resp_start:resp_end]
        label_window = labels[label_start:label_end]

        if len(resp_window) == 0 or len(label_window) == 0:
            continue

        values, counts = np.unique(label_window, return_counts=True)
        majority_index = int(np.argmax(counts))
        original_label = int(values[majority_index])
        label_purity = counts[majority_index] / len(label_window)

        if original_label not in VALID_ORIGINAL_LABELS:
            continue

        if label_purity < purity_threshold:
            continue

        all_features = calculate_resp_features(resp_window)
        selected_features = {
            feature: all_features[feature] for feature in RESP_FEATURE_COLUMNS
        }

        rows.append(
            {
                "subject": subject,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "window_sec": window_sec,
                "step_sec": step_sec,
                **selected_features,
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
        subject_df = create_resp_features_for_subject(subject_folder)

        if subject_df.empty:
            print(f"Warning: nessuna finestra valida per {subject_folder.name}.")
            continue

        print(f"{subject_folder.name}: {len(subject_df)} finestre valide")
        subject_dataframes.append(subject_df)

    if not subject_dataframes:
        raise ValueError(
            "Nessuna riga generata. Controlla che i file .pkl contengano Resp chest, "
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
    print("\nDistribuzione original_label:")
    print(df["original_label"].value_counts().sort_index())
    print(f"\nPurezza minima: {df['label_purity'].min()}")
    print(f"NaN totali: {int(df.isna().sum().sum())}")
    print("\nBreath rate mean:")
    print(df["resp_breath_rate_mean"].describe())
    print("\nDominant respiratory rate:")
    print(df["resp_dominant_rate_bpm"].describe())
    print("\nDistribuzione per subject e label:")
    print(pd.crosstab(df["subject"], df["label"]))


if __name__ == "__main__":
    main()
