from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


DATASET_FILE = Path("data_features") / "bvp_features_all_60s.csv"
RESULTS_DIR = Path("results")
PER_SUBJECT_FILE = RESULTS_DIR / "bvp_knn_loso_per_subject.csv"
SUMMARY_FILE = RESULTS_DIR / "bvp_knn_summary.txt"
PROGRESS_FILE = Path("PROGRESS.md")

FEATURE_COLS = [
    "bvp_mean",
    "bvp_std",
    "bvp_min",
    "bvp_max",
    "bvp_range",
]

ENHANCED_FEATURE_COLS = [
    "bvp_median",
    "bvp_iqr",
    "bvp_rms",
    "bvp_energy",
    "bvp_skew",
    "bvp_kurtosis",
]

PHYSIOLOGICAL_FEATURE_COLS = [
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

EXCLUDED_FEATURE_COLS = [
    "subject",
    "start_sec",
    "end_sec",
    "label",
    "original_label",
    "label_purity",
]

TARGET_COL = "label"
GROUP_COL = "subject"
K_NEIGHBORS = 9
KNN_WEIGHTS = "distance"
PAPER_ACCURACY = 0.8206
PAPER_F1 = 0.7894
LOW_PERFORMANCE_MARGIN = 0.15


def format_percent(value):
    return f"{value * 100:.2f}%"


def format_counts(series):
    counts = series.value_counts().sort_index()
    return "; ".join(f"{int(label)}:{int(count)}" for label, count in counts.items())


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_FILE}")

    return pd.read_csv(DATASET_FILE)


def validate_columns(df):
    required_columns = FEATURE_COLS + [TARGET_COL, GROUP_COL]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonne richieste mancanti: {missing_columns}")

    missing_excluded_columns = [
        column for column in EXCLUDED_FEATURE_COLS if column not in df.columns
    ]
    if missing_excluded_columns:
        raise ValueError(
            "Colonne metadata/target attese mancanti: "
            f"{missing_excluded_columns}"
        )

    if df.empty:
        raise ValueError("Dataset vuoto.")

    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 classi in y per addestrare kNN.")

    if df[GROUP_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 soggetti per Leave-One-Subject-Out.")

    selected_feature_cols = get_feature_columns(df)
    feature_nan_counts = df[selected_feature_cols].isna().sum()
    if feature_nan_counts.any():
        raise ValueError(
            "NaN nelle feature: "
            f"{feature_nan_counts[feature_nan_counts > 0].to_dict()}"
        )


def get_feature_columns(df):
    available_enhanced_cols = [
        column for column in ENHANCED_FEATURE_COLS if column in df.columns
    ]
    available_physiological_cols = [
        column for column in PHYSIOLOGICAL_FEATURE_COLS if column in df.columns
    ]
    return FEATURE_COLS + available_enhanced_cols + available_physiological_cols


def get_binary_confusion_values(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def run_loso_knn(df):
    selected_feature_cols = get_feature_columns(df)
    X = df[selected_feature_cols]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    logo = LeaveOneGroupOut()
    fold_rows = []
    all_true = []
    all_pred = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        left_out_subject = groups.iloc[test_idx].iloc[0]

        print(f"LOSO fold - soggetto lasciato fuori: {left_out_subject}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights=KNN_WEIGHTS)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

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

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    return pd.DataFrame(fold_rows), pd.Series(all_true), pd.Series(all_pred)


def build_summary(df, per_subject_df, y_true, y_pred):
    selected_feature_cols = get_feature_columns(df)
    missing_enhanced_cols = [
        column for column in ENHANCED_FEATURE_COLS if column not in df.columns
    ]
    missing_physiological_cols = [
        column for column in PHYSIOLOGICAL_FEATURE_COLS if column not in df.columns
    ]
    mean_metrics = per_subject_df[["accuracy", "precision", "recall", "f1"]].mean()
    aggregate_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = aggregate_cm.ravel()
    predicted_classes = [int(label) for label in sorted(pd.Series(y_pred).unique())]
    predicts_single_class = len(predicted_classes) == 1
    low_f1_threshold = max(mean_metrics["f1"] - LOW_PERFORMANCE_MARGIN, 0)
    low_performance_subjects = per_subject_df[
        per_subject_df["f1"] < low_f1_threshold
    ].copy()

    aggregate_accuracy = accuracy_score(y_true, y_pred)
    aggregate_precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )
    aggregate_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    aggregate_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    report = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)

    lines = [
        "# BVP wrist + kNN LOSO summary",
        "",
        f"Dataset usato: {DATASET_FILE}",
        f"Feature usate: {', '.join(selected_feature_cols)}",
        "Feature avanzate richieste ma non presenti nel CSV: "
        f"{', '.join(missing_enhanced_cols) if missing_enhanced_cols else 'nessuna'}",
        "Feature fisiologiche HR/HRV richieste ma non presenti nel CSV: "
        f"{', '.join(missing_physiological_cols) if missing_physiological_cols else 'nessuna'}",
        "Feature escluse per evitare leakage: "
        f"{', '.join(EXCLUDED_FEATURE_COLS)}",
        "Modello usato: KNeighborsClassifier",
        f"n_neighbors = {K_NEIGHBORS}",
        f"weights = {KNN_WEIGHTS}",
        "Metodo di validazione: Leave-One-Subject-Out",
        f"Numero totale di soggetti: {df[GROUP_COL].nunique()}",
        f"Numero totale di finestre/campioni: {len(df)}",
        f"Distribuzione globale delle classi: {format_counts(df[TARGET_COL])}",
        "",
        "## Metriche medie sui fold",
        f"Accuracy media sui fold: {mean_metrics['accuracy']:.4f} "
        f"({format_percent(mean_metrics['accuracy'])})",
        f"Precision media sui fold: {mean_metrics['precision']:.4f} "
        f"({format_percent(mean_metrics['precision'])})",
        f"Recall media sui fold: {mean_metrics['recall']:.4f} "
        f"({format_percent(mean_metrics['recall'])})",
        f"F1 media sui fold: {mean_metrics['f1']:.4f} "
        f"({format_percent(mean_metrics['f1'])})",
        "",
        "## Metriche aggregate globali",
        f"Accuracy aggregata globale: {aggregate_accuracy:.4f} "
        f"({format_percent(aggregate_accuracy)})",
        f"Precision aggregata globale: {aggregate_precision:.4f} "
        f"({format_percent(aggregate_precision)})",
        f"Recall aggregata globale: {aggregate_recall:.4f} "
        f"({format_percent(aggregate_recall)})",
        f"F1 aggregata globale: {aggregate_f1:.4f} "
        f"({format_percent(aggregate_f1)})",
        "",
        "## Confusion matrix aggregata",
        "Righe = classi vere, colonne = classi predette, ordine classi [0, 1].",
        str(aggregate_cm),
        f"TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}",
        f"Falsi stress (FP, non-stress predetto stress): {int(fp)}",
        f"Stress mancati (FN, stress predetto non-stress): {int(fn)}",
        "",
        "## Classification report aggregato",
        report,
        "",
        "## Controlli predizioni",
        f"Classi predette globalmente: {predicted_classes}",
    ]

    if predicts_single_class:
        lines.append("ATTENZIONE: il modello predice una sola classe.")
    else:
        lines.append("Il modello predice entrambe le classi.")

    lines.extend(
        [
            "",
            "## Soggetti con performance molto piu' bassa",
            f"Criterio: F1 < media fold F1 - {LOW_PERFORMANCE_MARGIN:.2f} "
            f"({low_f1_threshold:.4f}).",
        ]
    )

    if low_performance_subjects.empty:
        lines.append("Nessun soggetto sotto la soglia di attenzione.")
    else:
        for _, row in low_performance_subjects.iterrows():
            lines.append(
                f"- {row['left_out_subject']}: accuracy={row['accuracy']:.4f}, "
                f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
                f"f1={row['f1']:.4f}, TN={row['tn']}, FP={row['fp']}, "
                f"FN={row['fn']}, TP={row['tp']}"
            )

    lines.extend(
        [
            "",
            "## Confronto indicativo con il paper",
            f"Paper WESAD BVP + kNN accuracy circa: {format_percent(PAPER_ACCURACY)}",
            f"Paper WESAD BVP + kNN F1 circa: {format_percent(PAPER_F1)}",
        "Nota: il confronto e' indicativo perche' questa pipeline usa feature "
            "BVP statistiche e fisiologiche disponibili nel CSV.",
        ]
    )

    summary_text = "\n".join(lines) + "\n"
    summary_values = {
        "mean_accuracy": mean_metrics["accuracy"],
        "mean_precision": mean_metrics["precision"],
        "mean_recall": mean_metrics["recall"],
        "mean_f1": mean_metrics["f1"],
        "aggregate_accuracy": aggregate_accuracy,
        "aggregate_precision": aggregate_precision,
        "aggregate_recall": aggregate_recall,
        "aggregate_f1": aggregate_f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_classes": predicted_classes,
        "predicts_single_class": predicts_single_class,
        "selected_feature_cols": selected_feature_cols,
        "missing_enhanced_cols": missing_enhanced_cols,
        "missing_physiological_cols": missing_physiological_cols,
        "low_performance_subjects": low_performance_subjects[
            "left_out_subject"
        ].tolist(),
    }

    return summary_text, summary_values


def update_progress(summary_values):
    status = (
        "Il modello predice una sola classe."
        if summary_values["predicts_single_class"]
        else "Il modello predice entrambe le classi."
    )
    low_subjects = summary_values["low_performance_subjects"]
    low_subjects_text = (
        ", ".join(low_subjects) if low_subjects else "nessun soggetto sotto soglia"
    )

    section_title = "## Passo 7 - BVP wrist + kNN LOSO"
    section = f"""{section_title}

- Script creato: `src/train_bvp_knn_loso.py`
- Dataset usato: `data_features/bvp_features_all_60s.csv`
- CSV rigenerato da `WESAD/` con feature statistiche, peak detection e HR/HRV semplici.
- Feature usate: `{", ".join(summary_values["selected_feature_cols"])}`
- Feature avanzate mancanti nel CSV: `{", ".join(summary_values["missing_enhanced_cols"]) if summary_values["missing_enhanced_cols"] else "nessuna"}`
- Feature fisiologiche HR/HRV mancanti nel CSV: `{", ".join(summary_values["missing_physiological_cols"]) if summary_values["missing_physiological_cols"] else "nessuna"}`
- Validazione: Leave-One-Subject-Out per soggetto, con `StandardScaler` fittato solo sul training fold.
- Modello: `KNeighborsClassifier(n_neighbors=9, weights="distance")`
- Risultati principali:
  - Accuracy media fold: {summary_values["mean_accuracy"]:.4f} ({format_percent(summary_values["mean_accuracy"])})
  - F1 media fold: {summary_values["mean_f1"]:.4f} ({format_percent(summary_values["mean_f1"])})
  - Accuracy aggregata globale: {summary_values["aggregate_accuracy"]:.4f} ({format_percent(summary_values["aggregate_accuracy"])})
  - F1 aggregata globale: {summary_values["aggregate_f1"]:.4f} ({format_percent(summary_values["aggregate_f1"])})
  - Confusion matrix aggregata: TN={summary_values["tn"]}, FP={summary_values["fp"]}, FN={summary_values["fn"]}, TP={summary_values["tp"]}
- File generati:
  - `results/bvp_knn_loso_per_subject.csv`
  - `results/bvp_knn_summary.txt`
- Osservazioni:
  - {status}
  - Soggetti problematici secondo soglia F1: {low_subjects_text}
  - Confronto paper indicativo: BVP + kNN circa 82.06% accuracy e 78.94% F1.
"""

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


def save_results(df, per_subject_df, y_true, y_pred):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_subject_df.to_csv(PER_SUBJECT_FILE, index=False)

    summary_text, summary_values = build_summary(df, per_subject_df, y_true, y_pred)
    SUMMARY_FILE.write_text(summary_text, encoding="utf-8")
    update_progress(summary_values)

    return summary_values


def main():
    df = load_dataset()
    validate_columns(df)
    per_subject_df, y_true, y_pred = run_loso_knn(df)
    summary_values = save_results(df, per_subject_df, y_true, y_pred)

    print("\nRisultati salvati:")
    print(f"- {PER_SUBJECT_FILE}")
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
