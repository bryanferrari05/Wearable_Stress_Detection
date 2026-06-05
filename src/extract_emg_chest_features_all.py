import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch


DATA_RAW_DIR_CANDIDATES = (
    Path("data_raw"),
    Path("WESAD"),
    Path.home() / "Downloads" / "WESAD" / "WESAD",
    Path.home() / "Downloads" / "WESAD",
    Path("."),
)
OUTPUT_DIR = Path("data_features")
OUTPUT_FILE = OUTPUT_DIR / "emg_chest_features_all_60s.csv"

FS_EMG = 700
FS_LABEL = 700
WINDOW_SEC = 60
STEP_SEC = 30
PURITY_THRESHOLD = 0.70

LOWCUT_HZ = 20.0
HIGHCUT_HZ = 250.0
WELCH_NPERSEG = 4096
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


def filter_emg_signal(values):
    signal = np.asarray(values, dtype=float)
    sos = butter(
        4,
        [LOWCUT_HZ, HIGHCUT_HZ],
        btype="bandpass",
        fs=FS_EMG,
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def calculate_zero_crossing_count(signal):
    threshold = 0.01 * np.std(signal)
    left = signal[:-1]
    right = signal[1:]
    sign_changes = left * right < 0
    amplitude_changes = np.abs(right - left) >= threshold
    return int(np.sum(sign_changes & amplitude_changes))


def calculate_slope_sign_change_count(signal):
    threshold = 0.01 * np.std(signal)
    diff = np.diff(signal)

    if len(diff) < 2:
        return 0

    slope_changes = diff[:-1] * diff[1:] < 0
    amplitude_changes = (
        (np.abs(diff[:-1]) >= threshold) | (np.abs(diff[1:]) >= threshold)
    )
    return int(np.sum(slope_changes & amplitude_changes))


def calculate_hjorth_features(signal):
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)
    activity = np.var(signal)
    first_var = np.var(first_derivative) if len(first_derivative) else 0.0
    second_var = np.var(second_derivative) if len(second_derivative) else 0.0

    mobility = np.sqrt(first_var / activity) if activity > 0 else 0.0
    first_mobility = np.sqrt(second_var / first_var) if first_var > 0 else 0.0
    complexity = first_mobility / mobility if mobility > 0 else 0.0

    return {
        "emg_hjorth_activity": activity,
        "emg_hjorth_mobility": mobility,
        "emg_hjorth_complexity": complexity,
    }


def calculate_time_domain_emg_features(filtered_signal):
    signal = np.asarray(filtered_signal, dtype=float)
    rectified = np.abs(signal)
    derivative = np.diff(signal)
    waveform_length = np.sum(np.abs(derivative)) if len(derivative) else 0.0
    window_duration_sec = len(signal) / FS_EMG
    zero_crossing_count = calculate_zero_crossing_count(signal)
    slope_sign_change_count = calculate_slope_sign_change_count(signal)
    wamp_threshold = 0.05 * np.std(signal)
    willison_count = (
        int(np.sum(np.abs(derivative) >= wamp_threshold)) if len(derivative) else 0
    )

    features = {
        "emg_abs_mean": np.mean(rectified),
        "emg_abs_std": np.std(rectified),
        "emg_abs_median": np.median(rectified),
        "emg_abs_max": np.max(rectified),
        "emg_iemg": np.sum(rectified),
        "emg_mav": np.mean(rectified),
        "emg_log_detector": np.exp(np.mean(np.log(rectified + EPSILON))),
        "emg_variance": np.var(signal),
        "emg_waveform_length": waveform_length,
        "emg_average_amplitude_change": (
            waveform_length / len(derivative) if len(derivative) else 0.0
        ),
        "emg_derivative_mean": np.mean(derivative) if len(derivative) else 0.0,
        "emg_derivative_std": np.std(derivative) if len(derivative) else 0.0,
        "emg_derivative_max_abs": (
            np.max(np.abs(derivative)) if len(derivative) else 0.0
        ),
        "emg_zero_crossing_count": zero_crossing_count,
        "emg_zero_crossing_rate": (
            zero_crossing_count / window_duration_sec if window_duration_sec else 0.0
        ),
        "emg_slope_sign_change_count": slope_sign_change_count,
        "emg_slope_sign_change_rate": (
            slope_sign_change_count / window_duration_sec
            if window_duration_sec
            else 0.0
        ),
        "emg_willison_amplitude_count": willison_count,
        "emg_willison_amplitude_rate": (
            willison_count / window_duration_sec if window_duration_sec else 0.0
        ),
    }
    features.update(calculate_hjorth_features(signal))
    return features


def calculate_band_power(frequencies, power_spectrum, low_hz, high_hz):
    mask = (frequencies >= low_hz) & (frequencies < high_hz)

    if not np.any(mask):
        return 0.0

    return float(np.trapezoid(power_spectrum[mask], frequencies[mask]))


def calculate_frequency_domain_emg_features(filtered_signal):
    nperseg = min(WELCH_NPERSEG, len(filtered_signal))
    frequencies, power_spectrum = welch(
        filtered_signal,
        fs=FS_EMG,
        nperseg=nperseg,
    )
    frequency_mask = (frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ)
    emg_frequencies = frequencies[frequency_mask]
    emg_power_spectrum = power_spectrum[frequency_mask]

    if len(emg_frequencies) == 0:
        return {
            "emg_total_power": 0.0,
            "emg_mean_frequency": 0.0,
            "emg_median_frequency": 0.0,
            "emg_bandpower_20_60": 0.0,
            "emg_bandpower_60_120": 0.0,
            "emg_bandpower_120_250": 0.0,
            "emg_relative_power_20_60": 0.0,
            "emg_relative_power_60_120": 0.0,
            "emg_relative_power_120_250": 0.0,
        }

    total_power = float(np.trapezoid(emg_power_spectrum, emg_frequencies))
    weighted_power_sum = np.sum(emg_frequencies * emg_power_spectrum)
    power_sum = np.sum(emg_power_spectrum)
    cumulative_power = np.cumsum(emg_power_spectrum)

    if power_sum > 0:
        mean_frequency = weighted_power_sum / power_sum
        median_frequency = emg_frequencies[
            np.searchsorted(cumulative_power, cumulative_power[-1] / 2)
        ]
    else:
        mean_frequency = 0.0
        median_frequency = 0.0

    bandpower_20_60 = calculate_band_power(
        emg_frequencies, emg_power_spectrum, 20.0, 60.0
    )
    bandpower_60_120 = calculate_band_power(
        emg_frequencies, emg_power_spectrum, 60.0, 120.0
    )
    bandpower_120_250 = calculate_band_power(
        emg_frequencies, emg_power_spectrum, 120.0, 250.0
    )

    return {
        "emg_total_power": total_power,
        "emg_mean_frequency": mean_frequency,
        "emg_median_frequency": median_frequency,
        "emg_bandpower_20_60": bandpower_20_60,
        "emg_bandpower_60_120": bandpower_60_120,
        "emg_bandpower_120_250": bandpower_120_250,
        "emg_relative_power_20_60": (
            bandpower_20_60 / total_power if total_power > 0 else 0.0
        ),
        "emg_relative_power_60_120": (
            bandpower_60_120 / total_power if total_power > 0 else 0.0
        ),
        "emg_relative_power_120_250": (
            bandpower_120_250 / total_power if total_power > 0 else 0.0
        ),
    }


def calculate_emg_features(values):
    signal = np.asarray(values, dtype=float)
    filtered = filter_emg_signal(signal)

    features = calculate_signal_features(filtered, "emg")
    features.update(calculate_time_domain_emg_features(filtered))
    features.update(calculate_frequency_domain_emg_features(filtered))
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


def create_emg_features_for_subject(
    subject_folder,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
    purity_threshold=PURITY_THRESHOLD,
):
    data = load_subject_pickle(subject_folder)

    if data is None:
        return pd.DataFrame()

    try:
        emg = np.asarray(data["signal"]["chest"]["EMG"]).ravel()
        labels = np.asarray(data["label"]).ravel()
        subject = str(data.get("subject", subject_folder.name))
    except KeyError as error:
        subject = str(data.get("subject", subject_folder.name))
        print(f"Warning: salto {subject}, campo mancante: {error}")
        return pd.DataFrame()

    if len(emg) == 0 or len(labels) == 0:
        print(f"Warning: salto {subject}, EMG o label vuoti.")
        return pd.DataFrame()

    duration_sec = min(len(emg) / FS_EMG, len(labels) / FS_LABEL)
    max_start_sec = int(np.floor(duration_sec - window_sec))

    if max_start_sec < 0:
        print(f"Warning: salto {subject}, durata inferiore a {window_sec} secondi.")
        return pd.DataFrame()

    rows = []

    for start_sec in range(0, max_start_sec + 1, step_sec):
        end_sec = start_sec + window_sec

        emg_start = int(round(start_sec * FS_EMG))
        emg_end = int(round(end_sec * FS_EMG))
        label_start = int(round(start_sec * FS_LABEL))
        label_end = int(round(end_sec * FS_LABEL))

        emg_window = emg[emg_start:emg_end]
        label_window = labels[label_start:label_end]

        if len(emg_window) == 0 or len(label_window) == 0:
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
                **calculate_emg_features(emg_window),
                "original_label": original_label,
                "label": LABEL_MAPPING[original_label],
                "label_purity": label_purity,
            }
        )

    return pd.DataFrame(rows)


def main():
    subject_folders = find_subject_folders()
    subject_dataframes = []

    for subject_folder in subject_folders:
        subject_df = create_emg_features_for_subject(subject_folder)

        if subject_df.empty:
            print(f"Warning: nessuna finestra valida per {subject_folder.name}.")
            continue

        print(f"{subject_folder.name}: {len(subject_df)} finestre valide")
        subject_dataframes.append(subject_df)

    if not subject_dataframes:
        raise ValueError(
            "Nessuna riga generata. Controlla che i file .pkl contengano EMG chest, "
            "label e finestre con original_label in [1, 2, 3] e purezza >= 0.70."
        )

    df = pd.concat(subject_dataframes, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFile salvato: {OUTPUT_FILE}")

    print(f"\nShape DataFrame: {df.shape}")
    print(f"Numero soggetti: {df['subject'].nunique()}")
    print("\nDistribuzione label binarie:")
    print(df["label"].value_counts().sort_index())
    print("\nDistribuzione original_label:")
    print(df["original_label"].value_counts().sort_index())
    print(f"\nPurezza minima: {df['label_purity'].min()}")
    print(f"NaN totali: {int(df.isna().sum().sum())}")
    print("\nDistribuzione RMS EMG:")
    print(df["emg_rms"].describe())
    print("\nDistribuzione mean frequency EMG:")
    print(df["emg_mean_frequency"].describe())
    print("\nDistribuzione per subject e label:")
    print(pd.crosstab(df["subject"], df["label"]))


if __name__ == "__main__":
    main()
