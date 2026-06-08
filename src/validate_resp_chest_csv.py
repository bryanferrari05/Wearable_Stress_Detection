from pathlib import Path

import numpy as np
import pandas as pd


CSV_FILE = Path("data_features") / "resp_chest_features_all_60s.csv"
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

    resp_alignment = df[ALIGNMENT_COLUMNS].reset_index(drop=True)
    bvp_alignment = bvp_df[ALIGNMENT_COLUMNS].reset_index(drop=True)
    if not resp_alignment.equals(bvp_alignment):
        errors.append(
            "Finestre o label RESP chest non allineate al CSV BVP di riferimento."
        )


def validate_resp_chest_csv(csv_file=CSV_FILE):
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
        errors.append("Sono presenti NaN o valori infiniti nelle feature RESP chest.")

    if (df["end_sec"] - df["start_sec"] != df["window_sec"]).any():
        errors.append("Sono presenti finestre con durata non coerente.")
    non_negative_columns = [
        "resp_range",
        "resp_peak_rate_per_min",
        "resp_trough_rate_per_min",
        "resp_breath_rate_mean",
        "resp_breath_rate_std",
        "resp_breath_rate_range",
        "resp_interval_rmssd",
        "resp_interval_sdnn",
        "resp_cycle_amplitude_mean",
        "resp_cycle_amplitude_std",
    ]
    for column in non_negative_columns:
        if (df[column] < 0).any():
            errors.append(f"Sono presenti valori negativi in {column}.")

    if ((df["resp_dominant_rate_bpm"] < 0) | (df["resp_dominant_rate_bpm"] > 120)).any():
        errors.append("resp_dominant_rate_bpm fuori dal range atteso [0, 120].")

    relative_power_column = "resp_relative_power_0_75_2_00"
    if (
        (df[relative_power_column] < 0) | (df[relative_power_column] > 1)
    ).any():
        errors.append(f"Sono presenti valori {relative_power_column} fuori da [0, 1].")

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
    df, errors = validate_resp_chest_csv()
    print(f"CSV: {CSV_FILE}")
    print(f"Shape: {df.shape}")

    if not df.empty and all(column in df.columns for column in REQUIRED_COLUMNS):
        print(f"Soggetti ({df['subject'].nunique()}): {sorted(df['subject'].unique())}")
        print("\nDistribuzione label:")
        print(df["label"].value_counts().sort_index())
        print(f"\nPurezza minima: {df['label_purity'].min()}")
        print(f"NaN totali: {int(df.isna().sum().sum())}")
        print("\nBreath rate mean:")
        print(df["resp_breath_rate_mean"].describe())
        print("\nDominant respiratory rate:")
        print(df["resp_dominant_rate_bpm"].describe())

    if errors:
        print("\nVALIDAZIONE FALLITA")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDAZIONE OK")


if __name__ == "__main__":
    main()
