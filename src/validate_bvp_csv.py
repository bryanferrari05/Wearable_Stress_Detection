from pathlib import Path

import pandas as pd


CSV_FILE = Path("data_features") / "bvp_features_all_60s.csv"
PURITY_THRESHOLD = 0.70

EXCLUDED_SUBJECTS = {"S1", "S12"}
EXPECTED_SUBJECTS = {
    f"S{subject_id}"
    for subject_id in range(2, 18)
    if f"S{subject_id}" not in EXCLUDED_SUBJECTS
}

REQUIRED_COLUMNS = [
    "subject",
    "start_sec",
    "end_sec",
    "window_sec",
    "step_sec",
    "bvp_mean",
    "bvp_std",
    "bvp_min",
    "bvp_max",
    "bvp_range",
    "original_label",
    "label",
    "label_purity",
]

FEATURE_COLUMNS = [
    "bvp_mean",
    "bvp_std",
    "bvp_min",
    "bvp_max",
    "bvp_range",
]

OPTIONAL_ENHANCED_FEATURE_COLUMNS = [
    "bvp_median",
    "bvp_iqr",
    "bvp_rms",
    "bvp_energy",
    "bvp_skew",
    "bvp_kurtosis",
    "bvp_peak_count",
    "bvp_valid_ibi_count",
    "bvp_peak_prominence_mean",
    "bvp_peak_prominence_std",
    "bvp_ibi_mean",
    "bvp_ibi_std",
    "bvp_ibi_min",
    "bvp_ibi_max",
    "bvp_hr_mean",
    "bvp_hr_std",
    "bvp_hr_min",
    "bvp_hr_max",
    "bvp_rmssd",
    "bvp_sdnn",
]


def validate_bvp_csv(csv_file=CSV_FILE):
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

    if len(subjects) != len(EXPECTED_SUBJECTS):
        errors.append(
            f"Numero soggetti non atteso: {len(subjects)} "
            f"(attesi {len(EXPECTED_SUBJECTS)})."
        )

    if missing_subjects:
        errors.append(f"Soggetti attesi mancanti: {missing_subjects}")

    if unexpected_subjects:
        errors.append(f"Soggetti inattesi: {unexpected_subjects}")

    labels = set(df["label"].unique())
    if labels != {0, 1}:
        errors.append(f"Label binarie non valide: {sorted(labels)}")

    original_labels = set(df["original_label"].unique())
    if original_labels != {1, 2, 3}:
        errors.append(f"Original label non valide: {sorted(original_labels)}")

    min_purity = df["label_purity"].min()
    if min_purity < PURITY_THRESHOLD:
        errors.append(
            f"Purezza minima troppo bassa: {min_purity:.4f} "
            f"(< {PURITY_THRESHOLD:.2f})"
        )

    available_feature_columns = [
        column
        for column in FEATURE_COLUMNS + OPTIONAL_ENHANCED_FEATURE_COLUMNS
        if column in df.columns
    ]
    feature_nan_counts = df[available_feature_columns].isna().sum()
    if feature_nan_counts.any():
        errors.append(
            "NaN nelle feature: "
            f"{feature_nan_counts[feature_nan_counts > 0].to_dict()}"
        )

    invalid_window_rows = df[df["end_sec"] - df["start_sec"] != df["window_sec"]]
    if not invalid_window_rows.empty:
        errors.append(
            "Righe con end_sec - start_sec diverso da window_sec: "
            f"{len(invalid_window_rows)}"
        )

    invalid_range_rows = df[
        (df["bvp_max"] - df["bvp_min"] - df["bvp_range"]).abs() > 1e-9
    ]
    if not invalid_range_rows.empty:
        errors.append(
            "Righe con bvp_range diverso da bvp_max - bvp_min: "
            f"{len(invalid_range_rows)}"
        )

    subject_label_counts = df.groupby(["subject", "label"]).size().unstack(fill_value=0)
    subjects_without_both_labels = subject_label_counts[
        (subject_label_counts.get(0, 0) == 0) | (subject_label_counts.get(1, 0) == 0)
    ]
    if not subjects_without_both_labels.empty:
        errors.append(
            "Soggetti senza entrambe le classi: "
            f"{subjects_without_both_labels.index.tolist()}"
        )

    return df, errors


def main():
    df, errors = validate_bvp_csv()

    print(f"CSV: {CSV_FILE}")
    print(f"Shape: {df.shape}")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        print(f"Colonne presenti: {list(df.columns)}")
        print("\nVALIDAZIONE FALLITA")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Soggetti ({df['subject'].nunique()}): {sorted(df['subject'].unique())}")

    print("\nDistribuzione label:")
    print(df["label"].value_counts().sort_index())

    print("\nDistribuzione original_label:")
    print(df["original_label"].value_counts().sort_index())

    print("\nPurezza minima:")
    print(df["label_purity"].min())

    print("\nNaN per colonna:")
    print(df.isna().sum())

    print("\nDistribuzione per soggetto e label:")
    print(pd.crosstab(df["subject"], df["label"]))

    if errors:
        print("\nVALIDAZIONE FALLITA")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDAZIONE OK")


if __name__ == "__main__":
    main()
