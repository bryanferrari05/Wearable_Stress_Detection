from pathlib import Path

import numpy as np
import pandas as pd


CSV_FILE = Path("data_features") / "ecg_chest_features_all_60s.csv"
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
    "ecg_mean",
    "ecg_std",
    "ecg_min",
    "ecg_max",
    "ecg_range",
    "ecg_median",
    "ecg_iqr",
    "ecg_rms",
    "ecg_energy",
    "ecg_skew",
    "ecg_kurtosis",
    "ecg_derivative_mean",
    "ecg_derivative_std",
    "ecg_derivative_max_abs",
    "ecg_r_peak_count",
    "ecg_valid_rr_count",
    "ecg_r_peak_rate_per_min",
    "ecg_r_peak_prominence_mean",
    "ecg_r_peak_prominence_std",
    "ecg_rr_mean",
    "ecg_rr_std",
    "ecg_rr_min",
    "ecg_rr_max",
    "ecg_hr_mean",
    "ecg_hr_std",
    "ecg_hr_min",
    "ecg_hr_max",
    "ecg_hr_range",
    "ecg_rmssd",
    "ecg_sdnn",
    "ecg_nn50_count",
    "ecg_pnn50",
    "ecg_cvnn",
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

    ecg_alignment = df[ALIGNMENT_COLUMNS].reset_index(drop=True)
    bvp_alignment = bvp_df[ALIGNMENT_COLUMNS].reset_index(drop=True)
    if not ecg_alignment.equals(bvp_alignment):
        errors.append(
            "Finestre o label ECG chest non allineate al CSV BVP di riferimento."
        )


def validate_ecg_chest_csv(csv_file=CSV_FILE):
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
        errors.append("Sono presenti NaN o valori infiniti nelle feature ECG chest.")

    if (df["end_sec"] - df["start_sec"] != df["window_sec"]).any():
        errors.append("Sono presenti finestre con durata non coerente.")
    if ((df["ecg_max"] - df["ecg_min"] - df["ecg_range"]).abs() > 1e-9).any():
        errors.append("Sono presenti righe con ecg_range non coerente.")
    if (df["ecg_r_peak_count"] < 0).any():
        errors.append("Sono presenti conteggi R-peak negativi.")
    if (df["ecg_valid_rr_count"] < 0).any():
        errors.append("Sono presenti conteggi RR negativi.")
    if ((df["ecg_pnn50"] < 0) | (df["ecg_pnn50"] > 1)).any():
        errors.append("Sono presenti valori ecg_pnn50 fuori dall'intervallo [0, 1].")
    if (df["ecg_hr_min"] > df["ecg_hr_max"]).any():
        errors.append("Sono presenti righe con ecg_hr_min > ecg_hr_max.")

    expected_peak_rate = df["ecg_r_peak_count"] * 60 / df["window_sec"]
    if ((df["ecg_r_peak_rate_per_min"] - expected_peak_rate).abs() > 1e-9).any():
        errors.append("ecg_r_peak_rate_per_min non coerente con ecg_r_peak_count.")

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
    df, errors = validate_ecg_chest_csv()
    print(f"CSV: {CSV_FILE}")
    print(f"Shape: {df.shape}")

    if not df.empty and all(column in df.columns for column in REQUIRED_COLUMNS):
        print(f"Soggetti ({df['subject'].nunique()}): {sorted(df['subject'].unique())}")
        print("\nDistribuzione label:")
        print(df["label"].value_counts().sort_index())
        print(f"\nPurezza minima: {df['label_purity'].min()}")
        print(f"NaN totali: {int(df.isna().sum().sum())}")
        print("\nR-peak count:")
        print(df["ecg_r_peak_count"].describe())
        print("\nHR mean:")
        print(df["ecg_hr_mean"].describe())

    if errors:
        print("\nVALIDAZIONE FALLITA")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDAZIONE OK")


if __name__ == "__main__":
    main()
