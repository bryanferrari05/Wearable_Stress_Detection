from pathlib import Path

import numpy as np
import pandas as pd


CSV_FILE = Path("data_features") / "emg_chest_features_all_60s.csv"
BVP_REFERENCE_FILE = Path("data_features") / "bvp_features_all_60s.csv"
PURITY_THRESHOLD = 0.70

EXCLUDED_SUBJECTS = {"S1", "S12"}
EXPECTED_SUBJECTS = {
    f"S{subject_id}"
    for subject_id in range(2, 18)
    if f"S{subject_id}" not in EXCLUDED_SUBJECTS
}

METADATA_COLUMNS = [
    "subject",
    "start_sec",
    "end_sec",
    "window_sec",
    "step_sec",
    "original_label",
    "label",
    "label_purity",
]

FEATURE_COLUMNS = [
    "emg_mean",
    "emg_std",
    "emg_min",
    "emg_max",
    "emg_range",
    "emg_median",
    "emg_iqr",
    "emg_rms",
    "emg_energy",
    "emg_skew",
    "emg_kurtosis",
    "emg_abs_mean",
    "emg_abs_std",
    "emg_abs_median",
    "emg_abs_max",
    "emg_iemg",
    "emg_mav",
    "emg_log_detector",
    "emg_variance",
    "emg_waveform_length",
    "emg_average_amplitude_change",
    "emg_derivative_mean",
    "emg_derivative_std",
    "emg_derivative_max_abs",
    "emg_zero_crossing_count",
    "emg_zero_crossing_rate",
    "emg_slope_sign_change_count",
    "emg_slope_sign_change_rate",
    "emg_willison_amplitude_count",
    "emg_willison_amplitude_rate",
    "emg_hjorth_activity",
    "emg_hjorth_mobility",
    "emg_hjorth_complexity",
    "emg_total_power",
    "emg_mean_frequency",
    "emg_median_frequency",
    "emg_bandpower_20_60",
    "emg_bandpower_60_120",
    "emg_bandpower_120_250",
    "emg_relative_power_20_60",
    "emg_relative_power_60_120",
    "emg_relative_power_120_250",
]

REQUIRED_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS
ALIGNMENT_COLUMNS = [
    "subject",
    "start_sec",
    "end_sec",
    "window_sec",
    "step_sec",
    "original_label",
    "label",
    "label_purity",
]


def validate_alignment_with_bvp(df, errors):
    if not BVP_REFERENCE_FILE.exists():
        return

    bvp_df = pd.read_csv(BVP_REFERENCE_FILE)
    missing_bvp_columns = [
        column for column in ALIGNMENT_COLUMNS if column not in bvp_df.columns
    ]
    if missing_bvp_columns:
        errors.append(
            f"CSV BVP di riferimento senza colonne per confronto: {missing_bvp_columns}"
        )
        return

    emg_alignment = df[ALIGNMENT_COLUMNS].reset_index(drop=True)
    bvp_alignment = bvp_df[ALIGNMENT_COLUMNS].reset_index(drop=True)
    if not emg_alignment.equals(bvp_alignment):
        errors.append(
            "Finestre o label EMG chest non allineate al CSV BVP di riferimento."
        )


def validate_emg_chest_csv(csv_file=CSV_FILE):
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_file}")

    df = pd.read_csv(csv_file)
    errors = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        errors.append(f"Colonne mancanti: {missing_columns}")
        return df, errors

    if df.empty:
        errors.append("Il CSV e' vuoto.")
        return df, errors

    subjects = set(df["subject"].unique())
    missing_subjects = sorted(EXPECTED_SUBJECTS - subjects)
    unexpected_subjects = sorted(subjects - EXPECTED_SUBJECTS)

    if missing_subjects:
        errors.append(f"Soggetti attesi mancanti: {missing_subjects}")
    if unexpected_subjects:
        errors.append(f"Soggetti inattesi: {unexpected_subjects}")
    if set(df["label"].unique()) != {0, 1}:
        errors.append(f"Label binarie non valide: {sorted(df['label'].unique())}")
    if set(df["original_label"].unique()) != {1, 2, 3}:
        errors.append(
            f"Original label non valide: {sorted(df['original_label'].unique())}"
        )

    if df["label_purity"].min() < PURITY_THRESHOLD:
        errors.append(
            f"Purezza minima troppo bassa: {df['label_purity'].min():.4f} "
            f"(< {PURITY_THRESHOLD:.2f})"
        )

    finite_values = np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    if not finite_values.all():
        errors.append("Sono presenti NaN o valori infiniti nelle feature EMG chest.")

    if (df["end_sec"] - df["start_sec"] != df["window_sec"]).any():
        errors.append("Sono presenti finestre con durata non coerente.")
    if ((df["emg_max"] - df["emg_min"] - df["emg_range"]).abs() > 1e-9).any():
        errors.append("Sono presenti righe con emg_range non coerente.")

    count_columns = [
        "emg_zero_crossing_count",
        "emg_slope_sign_change_count",
        "emg_willison_amplitude_count",
    ]
    for column in count_columns:
        if (df[column] < 0).any():
            errors.append(f"Sono presenti conteggi negativi in {column}.")

    expected_zero_crossing_rate = df["emg_zero_crossing_count"] / df["window_sec"]
    if ((df["emg_zero_crossing_rate"] - expected_zero_crossing_rate).abs() > 1e-9).any():
        errors.append("emg_zero_crossing_rate non coerente con il conteggio.")

    expected_slope_rate = df["emg_slope_sign_change_count"] / df["window_sec"]
    if ((df["emg_slope_sign_change_rate"] - expected_slope_rate).abs() > 1e-9).any():
        errors.append("emg_slope_sign_change_rate non coerente con il conteggio.")

    expected_willison_rate = df["emg_willison_amplitude_count"] / df["window_sec"]
    if (
        (df["emg_willison_amplitude_rate"] - expected_willison_rate).abs() > 1e-9
    ).any():
        errors.append("emg_willison_amplitude_rate non coerente con il conteggio.")

    relative_power_columns = [
        "emg_relative_power_20_60",
        "emg_relative_power_60_120",
        "emg_relative_power_120_250",
    ]
    for column in relative_power_columns:
        if ((df[column] < 0) | (df[column] > 1)).any():
            errors.append(f"Sono presenti valori {column} fuori da [0, 1].")

    subject_label_counts = df.groupby(["subject", "label"]).size().unstack(fill_value=0)
    subjects_without_both_labels = subject_label_counts[
        (subject_label_counts.get(0, 0) == 0) | (subject_label_counts.get(1, 0) == 0)
    ]
    if not subjects_without_both_labels.empty:
        errors.append(
            "Soggetti senza entrambe le classi: "
            f"{subjects_without_both_labels.index.tolist()}"
        )

    validate_alignment_with_bvp(df, errors)
    return df, errors


def main():
    df, errors = validate_emg_chest_csv()
    print(f"CSV: {CSV_FILE}")
    print(f"Shape: {df.shape}")

    if not df.empty and all(column in df.columns for column in REQUIRED_COLUMNS):
        print(f"Soggetti ({df['subject'].nunique()}): {sorted(df['subject'].unique())}")
        print("\nDistribuzione label:")
        print(df["label"].value_counts().sort_index())
        print(f"\nPurezza minima: {df['label_purity'].min()}")
        print(f"NaN totali: {int(df.isna().sum().sum())}")
        print("\nRMS EMG:")
        print(df["emg_rms"].describe())
        print("\nMean frequency EMG:")
        print(df["emg_mean_frequency"].describe())

    if errors:
        print("\nVALIDAZIONE FALLITA")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDAZIONE OK")


if __name__ == "__main__":
    main()
