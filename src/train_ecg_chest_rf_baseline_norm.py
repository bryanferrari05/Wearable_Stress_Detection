from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut


DATASET_FILE = Path("data_features") / "ecg_chest_features_all_60s.csv"
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "ecg_chest_rf_baseline_norm_results.csv"
BEST_PER_SUBJECT_FILE = RESULTS_DIR / "ecg_chest_rf_baseline_norm_best_per_subject.csv"
BEST_FEATURE_IMPORTANCE_FILE = (
    RESULTS_DIR / "ecg_chest_rf_baseline_norm_best_feature_importance.csv"
)
SUMMARY_FILE = RESULTS_DIR / "ecg_chest_rf_baseline_norm_summary.txt"
PROGRESS_FILE = Path("PROGRESS.md")

TARGET_COL = "label"
GROUP_COL = "subject"
BASELINE_LABEL = 1
EPSILON = 1e-9

N_ESTIMATORS = 300
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2
CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42
LOW_PERFORMANCE_MARGIN = 0.15

ABSOLUTE_FEATURE_COLS = [
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

BASELINE_NORMALIZED_FEATURES = [
    "ecg_hr_mean",
    "ecg_hr_std",
    "ecg_hr_range",
    "ecg_rr_mean",
    "ecg_rr_std",
    "ecg_rmssd",
    "ecg_sdnn",
    "ecg_pnn50",
    "ecg_cvnn",
    "ecg_r_peak_rate_per_min",
    "ecg_energy",
    "ecg_kurtosis",
]

FEATURE_EXPLANATIONS = {
    "ecg_hr_mean": "frequenza cardiaca media nella finestra",
    "ecg_hr_std": "quanto varia la frequenza cardiaca nella finestra",
    "ecg_hr_range": "differenza tra frequenza cardiaca massima e minima",
    "ecg_rr_mean": "intervallo medio tra due battiti consecutivi",
    "ecg_rr_std": "variabilita' degli intervalli tra battiti",
    "ecg_rmssd": "variazione battito-battito degli intervalli RR",
    "ecg_sdnn": "variabilita' globale degli intervalli RR",
    "ecg_pnn50": "quota di variazioni RR maggiori di 50 ms",
    "ecg_cvnn": "variabilita' RR normalizzata rispetto al RR medio",
    "ecg_r_peak_rate_per_min": "numero di R-peak al minuto",
    "ecg_energy": "energia del segnale ECG nella finestra",
    "ecg_kurtosis": "forma della distribuzione del segnale ECG",
}

EXCLUDED_FEATURE_COLS = [
    "subject",
    "start_sec",
    "end_sec",
    "window_sec",
    "step_sec",
    "original_label",
    "label",
    "label_purity",
]

REFERENCE_BASELINE_RF = {
    "accuracy": 0.7484,
    "precision": 0.5760,
    "recall": 0.6137,
    "f1": 0.5943,
}


def format_percent(value):
    return f"{value * 100:.2f}%"


def format_counts(series):
    counts = series.value_counts().sort_index()
    return "; ".join(f"{int(label)}:{int(count)}" for label, count in counts.items())


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {DATASET_FILE}. "
            "Prima esegui src/extract_ecg_chest_features_all.py."
        )
    return pd.read_csv(DATASET_FILE)


def validate_dataset(df):
    required_columns = (
        ABSOLUTE_FEATURE_COLS
        + EXCLUDED_FEATURE_COLS
        + ["original_label"]
    )
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonne richieste mancanti: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset vuoto.")
    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 classi in y per addestrare Random Forest.")
    if df[GROUP_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 soggetti per Leave-One-Subject-Out.")
    if not np.isfinite(df[ABSOLUTE_FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("Sono presenti NaN o valori infiniti nelle feature ECG chest.")


def add_baseline_normalized_features(df):
    df = df.copy()
    baseline_df = df[df["original_label"] == BASELINE_LABEL]
    missing_subjects = sorted(set(df[GROUP_COL]) - set(baseline_df[GROUP_COL]))
    if missing_subjects:
        raise ValueError(f"Soggetti senza finestre baseline: {missing_subjects}")

    baseline_means = baseline_df.groupby(GROUP_COL)[
        BASELINE_NORMALIZED_FEATURES
    ].mean()
    baseline_stds = baseline_df.groupby(GROUP_COL)[
        BASELINE_NORMALIZED_FEATURES
    ].std(ddof=0)

    for feature in BASELINE_NORMALIZED_FEATURES:
        subject_mean = df[GROUP_COL].map(baseline_means[feature])
        subject_std = df[GROUP_COL].map(baseline_stds[feature]).replace(0, EPSILON)
        df[f"{feature}_baseline_delta"] = df[feature] - subject_mean
        df[f"{feature}_baseline_zscore"] = (df[feature] - subject_mean) / (
            subject_std + EPSILON
        )

    generated_cols = get_delta_features() + get_zscore_features()
    if not np.isfinite(df[generated_cols].to_numpy(dtype=float)).all():
        raise ValueError("Feature baseline-normalized ECG non finite.")

    return df


def get_delta_features():
    return [
        f"{feature}_baseline_delta" for feature in BASELINE_NORMALIZED_FEATURES
    ]


def get_zscore_features():
    return [
        f"{feature}_baseline_zscore" for feature in BASELINE_NORMALIZED_FEATURES
    ]


def get_feature_sets():
    delta_features = get_delta_features()
    zscore_features = get_zscore_features()
    return {
        "absolute_33": ABSOLUTE_FEATURE_COLS,
        "baseline_delta_12": delta_features,
        "baseline_zscore_12": zscore_features,
        "absolute_plus_delta_45": ABSOLUTE_FEATURE_COLS + delta_features,
        "absolute_plus_zscore_45": ABSOLUTE_FEATURE_COLS + zscore_features,
        "absolute_plus_delta_zscore_57": ABSOLUTE_FEATURE_COLS
        + delta_features
        + zscore_features,
    }


def build_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_binary_confusion_values(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def evaluate_feature_set(df, feature_set_name, feature_cols):
    X = df[feature_cols]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    logo = LeaveOneGroupOut()
    fold_rows = []
    all_true = []
    all_pred = []
    feature_importance_rows = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        left_out_subject = groups.iloc[test_idx].iloc[0]

        model = build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tn, fp, fn, tp = get_binary_confusion_values(y_test, y_pred)
        fold_rows.append(
            {
                "feature_set": feature_set_name,
                "left_out_subject": left_out_subject,
                "n_train_samples": len(train_idx),
                "n_test_samples": len(test_idx),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, pos_label=1, zero_division=0
                ),
                "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
                "true_class_counts": format_counts(y_test),
                "predicted_class_counts": format_counts(pd.Series(y_pred)),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

        for feature_name, importance in zip(feature_cols, model.feature_importances_):
            feature_importance_rows.append(
                {
                    "feature_set": feature_set_name,
                    "left_out_subject": left_out_subject,
                    "feature": feature_name,
                    "importance": importance,
                }
            )

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    y_true = pd.Series(all_true)
    y_pred = pd.Series(all_pred)
    fold_df = pd.DataFrame(fold_rows)
    tn, fp, fn, tp = get_binary_confusion_values(y_true, y_pred)
    predicted_classes = ",".join(str(int(label)) for label in sorted(y_pred.unique()))

    return {
        "summary": {
            "feature_set": feature_set_name,
            "n_features": len(feature_cols),
            "features": ",".join(feature_cols),
            "mean_accuracy": fold_df["accuracy"].mean(),
            "mean_precision": fold_df["precision"].mean(),
            "mean_recall": fold_df["recall"].mean(),
            "mean_f1": fold_df["f1"].mean(),
            "aggregate_accuracy": accuracy_score(y_true, y_pred),
            "aggregate_precision": precision_score(
                y_true, y_pred, pos_label=1, zero_division=0
            ),
            "aggregate_recall": recall_score(
                y_true, y_pred, pos_label=1, zero_division=0
            ),
            "aggregate_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "predicted_classes": predicted_classes,
            "predicts_single_class": len(set(y_pred)) == 1,
        },
        "fold_rows": fold_rows,
        "feature_importance_rows": feature_importance_rows,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_tests(df):
    summary_rows = []
    fold_rows = []
    feature_importance_rows = []
    predictions_by_feature_set = {}

    for feature_set_name, feature_cols in get_feature_sets().items():
        print(
            f"Test ECG baseline norm RF: {feature_set_name} "
            f"({len(feature_cols)} feature)"
        )
        result = evaluate_feature_set(df, feature_set_name, feature_cols)
        summary_rows.append(result["summary"])
        fold_rows.extend(result["fold_rows"])
        feature_importance_rows.extend(result["feature_importance_rows"])
        predictions_by_feature_set[feature_set_name] = (
            result["y_true"],
            result["y_pred"],
        )

    results_df = pd.DataFrame(summary_rows).sort_values(
        ["aggregate_f1", "aggregate_accuracy"], ascending=[False, False]
    )
    return (
        results_df.reset_index(drop=True),
        pd.DataFrame(fold_rows),
        pd.DataFrame(feature_importance_rows),
        predictions_by_feature_set,
    )


def aggregate_feature_importances(feature_importance_df, feature_set_name):
    selected = feature_importance_df[
        feature_importance_df["feature_set"] == feature_set_name
    ]
    return (
        selected.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_importance", "std": "std_importance"})
        .sort_values("mean_importance", ascending=False)
    )


def get_low_performance_subjects(best_per_subject_df):
    mean_f1 = best_per_subject_df["f1"].mean()
    low_f1_threshold = max(mean_f1 - LOW_PERFORMANCE_MARGIN, 0)
    low_rows = best_per_subject_df[
        best_per_subject_df["f1"] < low_f1_threshold
    ]
    return low_f1_threshold, low_rows


def build_feature_explanation_lines():
    lines = [
        "Per ciascuna feature selezionata sono state create due nuove colonne:",
        "- `_baseline_delta`: valore della finestra meno media baseline personale.",
        "- `_baseline_zscore`: delta diviso deviazione standard baseline personale.",
        "",
        "Feature normalizzate rispetto alla baseline:",
    ]
    for feature in BASELINE_NORMALIZED_FEATURES:
        lines.append(f"- `{feature}`: {FEATURE_EXPLANATIONS[feature]}.")
    return lines


def build_summary(
    df,
    results_df,
    best_per_subject_df,
    best_feature_importance_df,
    y_true_best,
    y_pred_best,
):
    best = results_df.iloc[0]
    absolute_row = results_df[results_df["feature_set"] == "absolute_33"].iloc[0]
    low_f1_threshold, low_rows = get_low_performance_subjects(best_per_subject_df)
    aggregate_cm = confusion_matrix(y_true_best, y_pred_best, labels=[0, 1])
    report = classification_report(
        y_true_best,
        y_pred_best,
        labels=[0, 1],
        zero_division=0,
    )

    lines = [
        "# ECG chest + Random Forest baseline normalization summary",
        "",
        f"Dataset usato: {DATASET_FILE}",
        "Modello: RandomForestClassifier",
        f"n_estimators = {N_ESTIMATORS}",
        f"max_depth = {MAX_DEPTH}",
        f"min_samples_leaf = {MIN_SAMPLES_LEAF}",
        f"class_weight = {CLASS_WEIGHT}",
        f"random_state = {RANDOM_STATE}",
        "Soglia di classificazione: 0.5, default di RandomForestClassifier.",
        "Validazione: Leave-One-Subject-Out per soggetto.",
        "Scaling: non usato, perche' Random Forest non richiede StandardScaler.",
        "Pulizia/scarto finestre: non applicata.",
        "",
        "## Assunzione metodologica",
        "La baseline personale e' stimata usando solo finestre con "
        "`original_label=1` dello stesso soggetto. Questo rappresenta una fase "
        "di calibrazione a riposo disponibile anche per il soggetto test. Non "
        "vengono usate label stress per costruire la baseline.",
        "",
        f"Numero soggetti: {df[GROUP_COL].nunique()}",
        f"Numero campioni: {len(df)}",
        f"Distribuzione classi: {format_counts(df[TARGET_COL])}",
        "",
        "## Feature aggiunte",
        *build_feature_explanation_lines(),
        "",
        "## Migliore configurazione",
        f"Feature set: {best['feature_set']}",
        f"Numero feature: {int(best['n_features'])}",
        f"Accuracy aggregata: {best['aggregate_accuracy']:.4f} "
        f"({format_percent(best['aggregate_accuracy'])})",
        f"Precision aggregata: {best['aggregate_precision']:.4f} "
        f"({format_percent(best['aggregate_precision'])})",
        f"Recall aggregata: {best['aggregate_recall']:.4f} "
        f"({format_percent(best['aggregate_recall'])})",
        f"F1 aggregato: {best['aggregate_f1']:.4f} "
        f"({format_percent(best['aggregate_f1'])})",
        f"Confusion matrix: TN={int(best['tn'])}, FP={int(best['fp'])}, "
        f"FN={int(best['fn'])}, TP={int(best['tp'])}",
        f"Classi predette: {best['predicted_classes']}",
        "",
        "## Confronto con baseline ECG assoluta",
        f"Baseline script precedente: F1={REFERENCE_BASELINE_RF['f1']:.4f} "
        f"({format_percent(REFERENCE_BASELINE_RF['f1'])}), "
        f"Accuracy={REFERENCE_BASELINE_RF['accuracy']:.4f} "
        f"({format_percent(REFERENCE_BASELINE_RF['accuracy'])})",
        f"Replica absolute_33 in questo script: F1={absolute_row['aggregate_f1']:.4f} "
        f"({format_percent(absolute_row['aggregate_f1'])}), "
        f"Accuracy={absolute_row['aggregate_accuracy']:.4f} "
        f"({format_percent(absolute_row['aggregate_accuracy'])})",
        f"Delta F1 migliore - baseline precedente: "
        f"{best['aggregate_f1'] - REFERENCE_BASELINE_RF['f1']:+.4f} "
        f"({format_percent(best['aggregate_f1'] - REFERENCE_BASELINE_RF['f1'])})",
        f"Delta accuracy migliore - baseline precedente: "
        f"{best['aggregate_accuracy'] - REFERENCE_BASELINE_RF['accuracy']:+.4f} "
        f"({format_percent(best['aggregate_accuracy'] - REFERENCE_BASELINE_RF['accuracy'])})",
        "",
        "## Confusion matrix migliore",
        "Righe = classi vere, colonne = classi predette, ordine classi [0, 1].",
        str(aggregate_cm),
        "",
        "## Classification report migliore",
        report,
        "",
        "## Tutti i feature set testati",
        results_df[
            [
                "feature_set",
                "n_features",
                "aggregate_accuracy",
                "aggregate_precision",
                "aggregate_recall",
                "aggregate_f1",
                "tn",
                "fp",
                "fn",
                "tp",
                "predicted_classes",
            ]
        ].to_string(index=False),
        "",
        "## Top feature della migliore configurazione",
        best_feature_importance_df.head(20).to_string(index=False),
        "",
        "## Soggetti con performance molto piu' bassa",
        f"Criterio: F1 < media fold F1 - {LOW_PERFORMANCE_MARGIN:.2f} "
        f"({low_f1_threshold:.4f}).",
    ]

    if low_rows.empty:
        lines.append("Nessun soggetto sotto la soglia di attenzione.")
    else:
        for _, row in low_rows.iterrows():
            lines.append(
                f"- {row['left_out_subject']}: accuracy={row['accuracy']:.4f}, "
                f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
                f"f1={row['f1']:.4f}, TN={row['tn']}, FP={row['fp']}, "
                f"FN={row['fn']}, TP={row['tp']}"
            )

    lines.extend(
        [
            "",
            "## Interpretazione",
        ]
    )
    if best["aggregate_f1"] > REFERENCE_BASELINE_RF["f1"]:
        lines.append(
            "La normalizzazione rispetto alla baseline personale migliora il F1 "
            "rispetto alla baseline ECG assoluta. Il risultato suggerisce che "
            "per ECG e HRV conta molto misurare lo scostamento dal riposo "
            "individuale, non solo il valore assoluto."
        )
    else:
        lines.append(
            "La normalizzazione rispetto alla baseline personale non migliora il F1 "
            "rispetto alla baseline ECG assoluta. In questa configurazione la "
            "Random Forest usa meglio le feature assolute."
        )

    return "\n".join(lines) + "\n"


def replace_or_append_progress_section(section_title, section):
    if not PROGRESS_FILE.exists():
        PROGRESS_FILE.write_text(f"# PROGRESS\n\n{section}", encoding="utf-8")
        return

    content = PROGRESS_FILE.read_text(encoding="utf-8")
    if section_title not in content:
        separator = "" if content.endswith("\n\n") else "\n\n"
        PROGRESS_FILE.write_text(f"{content}{separator}{section}", encoding="utf-8")
        return

    start = content.index(section_title)
    next_section = content.find("\n## ", start + len(section_title))
    if next_section == -1:
        updated_content = f"{content[:start]}{section}"
    else:
        updated_content = f"{content[:start]}{section}\n{content[next_section + 1:]}"

    PROGRESS_FILE.write_text(updated_content, encoding="utf-8")


def update_progress(results_df):
    best = results_df.iloc[0]
    section_title = "## Passo 12 - ECG chest RF baseline normalization"
    section = f"""{section_title}

- Script creato: `src/train_ecg_chest_rf_baseline_norm.py`
- Dataset usato: `data_features/ecg_chest_features_all_60s.csv`
- Assunzione: baseline personale disponibile per ogni soggetto, usando solo finestre con `original_label=1`.
- Modello: `RandomForestClassifier(n_estimators={N_ESTIMATORS}, min_samples_leaf={MIN_SAMPLES_LEAF}, class_weight="{CLASS_WEIGHT}", random_state={RANDOM_STATE})`
- Soglia di classificazione: `0.5`.
- Pulizia/scarto finestre: non applicata.
- Feature set testati: `absolute_33`, `baseline_delta_12`, `baseline_zscore_12`, `absolute_plus_delta_45`, `absolute_plus_zscore_45`, `absolute_plus_delta_zscore_57`.
- Migliore feature set: `{best["feature_set"]}` con `{int(best["n_features"])}` feature.
- Risultati migliori:
  - Accuracy aggregata: {best["aggregate_accuracy"]:.4f} ({format_percent(best["aggregate_accuracy"])})
  - Precision aggregata: {best["aggregate_precision"]:.4f} ({format_percent(best["aggregate_precision"])})
  - Recall aggregata: {best["aggregate_recall"]:.4f} ({format_percent(best["aggregate_recall"])})
  - F1 aggregato: {best["aggregate_f1"]:.4f} ({format_percent(best["aggregate_f1"])})
  - Confusion matrix: TN={int(best["tn"])}, FP={int(best["fp"])}, FN={int(best["fn"])}, TP={int(best["tp"])}
- Delta F1 vs baseline ECG RF precedente: {best["aggregate_f1"] - REFERENCE_BASELINE_RF["f1"]:+.4f} ({format_percent(best["aggregate_f1"] - REFERENCE_BASELINE_RF["f1"])})
- File generati:
  - `results/ecg_chest_rf_baseline_norm_results.csv`
  - `results/ecg_chest_rf_baseline_norm_best_per_subject.csv`
  - `results/ecg_chest_rf_baseline_norm_best_feature_importance.csv`
  - `results/ecg_chest_rf_baseline_norm_summary.txt`
"""
    replace_or_append_progress_section(section_title, section)


def save_results(
    df,
    results_df,
    fold_df,
    feature_importance_df,
    predictions_by_feature_set,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_FILE, index=False)

    best_feature_set = results_df.iloc[0]["feature_set"]
    best_per_subject_df = fold_df[fold_df["feature_set"] == best_feature_set].copy()
    best_per_subject_df.to_csv(BEST_PER_SUBJECT_FILE, index=False)

    best_feature_importance_df = aggregate_feature_importances(
        feature_importance_df,
        best_feature_set,
    )
    best_feature_importance_df.to_csv(BEST_FEATURE_IMPORTANCE_FILE, index=False)

    y_true_best, y_pred_best = predictions_by_feature_set[best_feature_set]
    SUMMARY_FILE.write_text(
        build_summary(
            df,
            results_df,
            best_per_subject_df,
            best_feature_importance_df,
            y_true_best,
            y_pred_best,
        ),
        encoding="utf-8",
    )
    update_progress(results_df)


def main():
    df = load_dataset()
    validate_dataset(df)
    df = add_baseline_normalized_features(df)
    results_df, fold_df, feature_importance_df, predictions_by_feature_set = run_tests(
        df
    )
    save_results(
        df,
        results_df,
        fold_df,
        feature_importance_df,
        predictions_by_feature_set,
    )

    best = results_df.iloc[0]
    print("\nTest ECG baseline normalization completato.")
    print(f"File risultati: {RESULTS_FILE}")
    print(f"File summary: {SUMMARY_FILE}")
    print(
        "Migliore configurazione: "
        f"{best['feature_set']}, "
        f"F1={best['aggregate_f1']:.4f}, "
        f"Accuracy={best['aggregate_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
