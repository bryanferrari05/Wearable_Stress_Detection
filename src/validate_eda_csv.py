from pathlib import Path

import numpy as np
import pandas as pd


CSV_FILE = Path("data_features") / "eda_features_all_60s.csv"
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
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "eda_range",
    "eda_median",
    "eda_iqr",
    "eda_rms",
    "eda_energy",
    "eda_skew",
    "eda_kurtosis",
    "eda_slope",
    "eda_derivative_mean",
    "eda_derivative_std",
    "eda_derivative_max",
    "eda_tonic_mean",
    "eda_tonic_std",
    "eda_tonic_slope",
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_phasic_max",
    "eda_phasic_auc",
    "eda_scr_count",
    "eda_scr_rate_per_min",
    "eda_scr_amplitude_mean",
    "eda_scr_amplitude_std",
    "eda_scr_amplitude_max",
    "eda_scr_prominence_mean",
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

    eda_alignment = df[ALIGNMENT_COLUMNS].reset_index(drop=True)
    bvp_alignment = bvp_df[ALIGNMENT_COLUMNS].reset_index(drop=True)
    if not eda_alignment.equals(bvp_alignment):
        errors.append(
            "Finestre o label EDA non allineate al CSV BVP di riferimento."
        )


def validate_eda_csv(csv_file=CSV_FILE):
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
        errors.append("Sono presenti NaN o valori infiniti nelle feature EDA.")

    if (df["end_sec"] - df["start_sec"] != df["window_sec"]).any():
        errors.append("Sono presenti finestre con durata non coerente.")
    if ((df["eda_max"] - df["eda_min"] - df["eda_range"]).abs() > 1e-9).any():
        errors.append("Sono presenti righe con eda_range non coerente.")
    if (df["eda_scr_count"] < 0).any():
        errors.append("Sono presenti conteggi SCR negativi.")
    if (df["eda_scr_amplitude_max"] < 0).any():
        errors.append("Sono presenti ampiezze SCR negative.")

    expected_scr_rate = df["eda_scr_count"] * 60 / df["window_sec"]
    if ((df["eda_scr_rate_per_min"] - expected_scr_rate).abs() > 1e-9).any():
        errors.append("eda_scr_rate_per_min non coerente con eda_scr_count.")

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
    df, errors = validate_eda_csv()
    print(f"CSV: {CSV_FILE}")
    print(f"Shape: {df.shape}")

    if not df.empty and all(column in df.columns for column in REQUIRED_COLUMNS):
        print(f"Soggetti ({df['subject'].nunique()}): {sorted(df['subject'].unique())}")
        print("\nDistribuzione label:")
        print(df["label"].value_counts().sort_index())
        print(f"\nPurezza minima: {df['label_purity'].min()}")
        print(f"NaN totali: {int(df.isna().sum().sum())}")
        print("\nSCR count:")
        print(df["eda_scr_count"].describe())

    if errors:
        print("\nVALIDAZIONE FALLITA")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDAZIONE OK")


if __name__ == "__main__":
    main()
