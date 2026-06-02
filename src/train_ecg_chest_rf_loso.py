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
PER_SUBJECT_FILE = RESULTS_DIR / "ecg_chest_rf_loso_per_subject.csv"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "ecg_chest_rf_feature_importance.csv"
SUMMARY_FILE = RESULTS_DIR / "ecg_chest_rf_summary.txt"
PROGRESS_FILE = Path("PROGRESS.md")

FEATURE_COLS = [
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

TARGET_COL = "label"
GROUP_COL = "subject"
N_ESTIMATORS = 300
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2
CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42
LOW_PERFORMANCE_MARGIN = 0.15


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
    required_columns = FEATURE_COLS + EXCLUDED_FEATURE_COLS
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonne richieste mancanti: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset vuoto.")
    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 classi in y per addestrare Random Forest.")
    if df[GROUP_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 soggetti per Leave-One-Subject-Out.")

    finite_features = np.isfinite(df[FEATURE_COLS].to_numpy(dtype=float))
    if not finite_features.all():
        raise ValueError("Sono presenti NaN o valori infiniti nelle feature ECG chest.")


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


def run_loso_random_forest(df):
    X = df[FEATURE_COLS]
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

        print(f"LOSO fold - soggetto lasciato fuori: {left_out_subject}")

        model = build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tn, fp, fn, tp = get_binary_confusion_values(y_test, y_pred)
        fold_rows.append(
            {
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

        for feature_name, importance in zip(FEATURE_COLS, model.feature_importances_):
            feature_importance_rows.append(
                {
                    "left_out_subject": left_out_subject,
                    "feature": feature_name,
                    "importance": importance,
                }
            )

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    return (
        pd.DataFrame(fold_rows),
        pd.Series(all_true),
        pd.Series(all_pred),
        pd.DataFrame(feature_importance_rows),
    )


def aggregate_feature_importances(feature_importance_df):
    return (
        feature_importance_df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_importance", "std": "std_importance"})
        .sort_values("mean_importance", ascending=False)
    )


def calculate_summary_values(df, per_subject_df, y_true, y_pred):
    mean_metrics = per_subject_df[["accuracy", "precision", "recall", "f1"]].mean()
    tn, fp, fn, tp = get_binary_confusion_values(y_true, y_pred)
    predicted_classes = [int(label) for label in sorted(y_pred.unique())]
    low_f1_threshold = max(mean_metrics["f1"] - LOW_PERFORMANCE_MARGIN, 0)
    low_performance_subjects = per_subject_df[
        per_subject_df["f1"] < low_f1_threshold
    ]["left_out_subject"].tolist()

    return {
        "n_subjects": df[GROUP_COL].nunique(),
        "n_samples": len(df),
        "mean_accuracy": mean_metrics["accuracy"],
        "mean_precision": mean_metrics["precision"],
        "mean_recall": mean_metrics["recall"],
        "mean_f1": mean_metrics["f1"],
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
        "predicts_single_class": len(predicted_classes) == 1,
        "low_f1_threshold": low_f1_threshold,
        "low_performance_subjects": low_performance_subjects,
    }


def build_summary(
    df,
    per_subject_df,
    feature_importance_summary_df,
    y_true,
    y_pred,
    summary_values,
):
    aggregate_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)
    low_rows = per_subject_df[
        per_subject_df["left_out_subject"].isin(
            summary_values["low_performance_subjects"]
        )
    ]

    lines = [
        "# ECG chest + Random Forest LOSO summary",
        "",
        f"Dataset usato: {DATASET_FILE}",
        f"Feature usate ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}",
        f"Feature escluse per evitare leakage: {', '.join(EXCLUDED_FEATURE_COLS)}",
        "Modello usato: RandomForestClassifier",
        f"n_estimators = {N_ESTIMATORS}",
        f"max_depth = {MAX_DEPTH}",
        f"min_samples_leaf = {MIN_SAMPLES_LEAF}",
        f"class_weight = {CLASS_WEIGHT}",
        f"random_state = {RANDOM_STATE}",
        "Metodo di validazione: Leave-One-Subject-Out",
        "Scaling: non usato, perche' Random Forest non richiede StandardScaler.",
        f"Numero totale di soggetti: {summary_values['n_subjects']}",
        f"Numero totale di finestre/campioni: {summary_values['n_samples']}",
        f"Distribuzione globale delle classi: {format_counts(df[TARGET_COL])}",
        "",
        "## Metriche medie sui fold",
        f"Accuracy media sui fold: {summary_values['mean_accuracy']:.4f} "
        f"({format_percent(summary_values['mean_accuracy'])})",
        f"Precision media sui fold: {summary_values['mean_precision']:.4f} "
        f"({format_percent(summary_values['mean_precision'])})",
        f"Recall media sui fold: {summary_values['mean_recall']:.4f} "
        f"({format_percent(summary_values['mean_recall'])})",
        f"F1 media sui fold: {summary_values['mean_f1']:.4f} "
        f"({format_percent(summary_values['mean_f1'])})",
        "",
        "## Metriche aggregate globali",
        f"Accuracy aggregata globale: {summary_values['aggregate_accuracy']:.4f} "
        f"({format_percent(summary_values['aggregate_accuracy'])})",
        f"Precision aggregata globale: {summary_values['aggregate_precision']:.4f} "
        f"({format_percent(summary_values['aggregate_precision'])})",
        f"Recall aggregata globale: {summary_values['aggregate_recall']:.4f} "
        f"({format_percent(summary_values['aggregate_recall'])})",
        f"F1 aggregata globale: {summary_values['aggregate_f1']:.4f} "
        f"({format_percent(summary_values['aggregate_f1'])})",
        "",
        "## Confusion matrix aggregata",
        "Righe = classi vere, colonne = classi predette, ordine classi [0, 1].",
        str(aggregate_cm),
        f"TN={summary_values['tn']}, FP={summary_values['fp']}, "
        f"FN={summary_values['fn']}, TP={summary_values['tp']}",
        "",
        "## Classification report aggregato",
        report,
        "",
        "## Controlli predizioni",
        f"Classi predette globalmente: {summary_values['predicted_classes']}",
        (
            "ATTENZIONE: il modello predice una sola classe."
            if summary_values["predicts_single_class"]
            else "Il modello predice entrambe le classi."
        ),
        "",
        "## Top feature Random Forest",
        feature_importance_summary_df.head(15).to_string(index=False),
        "",
        "## Soggetti con performance molto piu' bassa",
        f"Criterio: F1 < media fold F1 - {LOW_PERFORMANCE_MARGIN:.2f} "
        f"({summary_values['low_f1_threshold']:.4f}).",
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
            "## Nota",
            "Questa e' la baseline ECG chest con Random Forest. Il modello usa "
            "solo feature estratte dal segnale ECG chest e non include ancora "
            "tuning degli iperparametri.",
        ]
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


def update_progress(summary_values):
    low_subjects = summary_values["low_performance_subjects"]
    low_subjects_text = (
        ", ".join(low_subjects) if low_subjects else "nessun soggetto sotto soglia"
    )
    status = (
        "Il modello predice una sola classe."
        if summary_values["predicts_single_class"]
        else "Il modello predice entrambe le classi."
    )
    section_title = "## Passo 11 - ECG chest + Random Forest LOSO"
    section = f"""{section_title}

- Script creato: `src/train_ecg_chest_rf_loso.py`
- Dataset usato: `data_features/ecg_chest_features_all_60s.csv`
- Feature usate: `{", ".join(FEATURE_COLS)}`
- Validazione: Leave-One-Subject-Out per soggetto.
- Modello baseline: `RandomForestClassifier(n_estimators={N_ESTIMATORS}, min_samples_leaf={MIN_SAMPLES_LEAF}, class_weight="{CLASS_WEIGHT}", random_state={RANDOM_STATE})`
- Scaling: non usato, perche' Random Forest non richiede `StandardScaler`.
- Risultati principali:
  - Accuracy media fold: {summary_values["mean_accuracy"]:.4f} ({format_percent(summary_values["mean_accuracy"])})
  - F1 media fold: {summary_values["mean_f1"]:.4f} ({format_percent(summary_values["mean_f1"])})
  - Accuracy aggregata globale: {summary_values["aggregate_accuracy"]:.4f} ({format_percent(summary_values["aggregate_accuracy"])})
  - F1 aggregata globale: {summary_values["aggregate_f1"]:.4f} ({format_percent(summary_values["aggregate_f1"])})
  - Confusion matrix aggregata: TN={summary_values["tn"]}, FP={summary_values["fp"]}, FN={summary_values["fn"]}, TP={summary_values["tp"]}
- File generati:
  - `results/ecg_chest_rf_loso_per_subject.csv`
  - `results/ecg_chest_rf_feature_importance.csv`
  - `results/ecg_chest_rf_summary.txt`
- Osservazioni:
  - {status}
  - Soggetti problematici secondo soglia F1: {low_subjects_text}
  - Questa esecuzione e' la baseline Random Forest ECG chest; non include ancora tuning degli iperparametri.
"""
    replace_or_append_progress_section(section_title, section)


def save_results(df, per_subject_df, feature_importance_df, y_true, y_pred):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_subject_df.to_csv(PER_SUBJECT_FILE, index=False)

    feature_importance_summary_df = aggregate_feature_importances(feature_importance_df)
    feature_importance_summary_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    summary_values = calculate_summary_values(df, per_subject_df, y_true, y_pred)
    summary_text = build_summary(
        df,
        per_subject_df,
        feature_importance_summary_df,
        y_true,
        y_pred,
        summary_values,
    )
    SUMMARY_FILE.write_text(summary_text, encoding="utf-8")
    update_progress(summary_values)
    return summary_values


def main():
    df = load_dataset()
    validate_dataset(df)
    per_subject_df, y_true, y_pred, feature_importance_df = run_loso_random_forest(df)
    summary_values = save_results(
        df, per_subject_df, feature_importance_df, y_true, y_pred
    )

    print("\nRisultati salvati:")
    print(f"- {PER_SUBJECT_FILE}")
    print(f"- {FEATURE_IMPORTANCE_FILE}")
    print(f"- {SUMMARY_FILE}")
    print(f"- {PROGRESS_FILE}")
    print("\nMetriche aggregate:")
    print(
        f"Accuracy={summary_values['aggregate_accuracy']:.4f}, "
        f"Precision={summary_values['aggregate_precision']:.4f}, "
        f"Recall={summary_values['aggregate_recall']:.4f}, "
        f"F1={summary_values['aggregate_f1']:.4f}"
    )
    if summary_values["predicts_single_class"]:
        print("Attenzione: il modello predice una sola classe.")
    else:
        print("Il modello predice entrambe le classi.")


if __name__ == "__main__":
    main()
