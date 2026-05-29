from pathlib import Path

import numpy as np
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


DATASET_FILE = Path("data_features") / "eda_features_all_60s.csv"
RESULTS_DIR = Path("results")
PER_SUBJECT_FILE = RESULTS_DIR / "eda_knn_loso_reduced_per_subject.csv"
SUMMARY_FILE = RESULTS_DIR / "eda_knn_reduced_summary.txt"
PROGRESS_FILE = Path("PROGRESS.md")

BASELINE_AGGREGATE_ACCURACY = 0.8559
BASELINE_AGGREGATE_PRECISION = 0.7720
BASELINE_AGGREGATE_RECALL = 0.7383
BASELINE_AGGREGATE_F1 = 0.7548

FULL_FEATURE_COLS = [
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

FEATURE_COLS = [
    "eda_mean",
    "eda_std",
    "eda_range",
    "eda_skew",
    "eda_kurtosis",
    "eda_slope",
    "eda_derivative_mean",
    "eda_derivative_std",
    "eda_derivative_max",
    "eda_tonic_std",
    "eda_phasic_std",
    "eda_phasic_auc",
    "eda_scr_count",
    "eda_scr_amplitude_std",
    "eda_scr_amplitude_max",
    "eda_scr_prominence_mean",
]

REMOVED_FEATURES = {
    "eda_min": "ridondante con livello EDA globale e max/mean",
    "eda_max": "ridondante con livello EDA globale e range",
    "eda_median": "quasi identica a eda_mean/eda_tonic_mean",
    "eda_iqr": "molto correlata con eda_std/eda_tonic_std",
    "eda_rms": "quasi identica a eda_mean su questo CSV",
    "eda_energy": "derivata da RMS con finestra di lunghezza fissa",
    "eda_tonic_mean": "quasi identica a eda_mean",
    "eda_tonic_slope": "quasi identica a eda_slope",
    "eda_phasic_mean": "copia scalata di eda_phasic_auc con finestre fisse",
    "eda_phasic_max": "molto correlata con eda_scr_amplitude_max",
    "eda_scr_rate_per_min": "identica a eda_scr_count con finestre da 60 secondi",
    "eda_scr_amplitude_mean": "molto correlata con eda_scr_prominence_mean",
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

TARGET_COL = "label"
GROUP_COL = "subject"
K_NEIGHBORS = 9
KNN_WEIGHTS = "distance"
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


def validate_dataset(df):
    required_columns = FULL_FEATURE_COLS + EXCLUDED_FEATURE_COLS
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonne richieste mancanti: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset vuoto.")
    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 classi in y per addestrare kNN.")
    if df[GROUP_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 soggetti per Leave-One-Subject-Out.")

    finite_features = np.isfinite(df[FEATURE_COLS].to_numpy(dtype=float))
    if not finite_features.all():
        raise ValueError("Sono presenti NaN o valori infiniti nelle feature EDA ridotte.")


def get_binary_confusion_values(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def run_loso_knn(df):
    X = df[FEATURE_COLS]
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

        print(f"LOSO fold ridotto - soggetto lasciato fuori: {left_out_subject}")

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


def calculate_summary_values(df, per_subject_df, y_true, y_pred):
    mean_metrics = per_subject_df[["accuracy", "precision", "recall", "f1"]].mean()
    tn, fp, fn, tp = get_binary_confusion_values(y_true, y_pred)
    predicted_classes = [int(label) for label in sorted(y_pred.unique())]
    low_f1_threshold = max(mean_metrics["f1"] - LOW_PERFORMANCE_MARGIN, 0)
    low_performance_subjects = per_subject_df[
        per_subject_df["f1"] < low_f1_threshold
    ]["left_out_subject"].tolist()

    aggregate_accuracy = accuracy_score(y_true, y_pred)
    aggregate_precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )
    aggregate_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    aggregate_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "n_subjects": df[GROUP_COL].nunique(),
        "n_samples": len(df),
        "mean_accuracy": mean_metrics["accuracy"],
        "mean_precision": mean_metrics["precision"],
        "mean_recall": mean_metrics["recall"],
        "mean_f1": mean_metrics["f1"],
        "aggregate_accuracy": aggregate_accuracy,
        "aggregate_precision": aggregate_precision,
        "aggregate_recall": aggregate_recall,
        "aggregate_f1": aggregate_f1,
        "delta_accuracy": aggregate_accuracy - BASELINE_AGGREGATE_ACCURACY,
        "delta_precision": aggregate_precision - BASELINE_AGGREGATE_PRECISION,
        "delta_recall": aggregate_recall - BASELINE_AGGREGATE_RECALL,
        "delta_f1": aggregate_f1 - BASELINE_AGGREGATE_F1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "predicted_classes": predicted_classes,
        "predicts_single_class": len(predicted_classes) == 1,
        "low_f1_threshold": low_f1_threshold,
        "low_performance_subjects": low_performance_subjects,
    }


def build_summary(df, per_subject_df, y_true, y_pred, summary_values):
    aggregate_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)
    low_rows = per_subject_df[
        per_subject_df["left_out_subject"].isin(
            summary_values["low_performance_subjects"]
        )
    ]

    lines = [
        "# EDA wrist + kNN LOSO reduced-feature summary",
        "",
        f"Dataset usato: {DATASET_FILE}",
        f"Feature baseline complete ({len(FULL_FEATURE_COLS)}): "
        f"{', '.join(FULL_FEATURE_COLS)}",
        f"Feature ridotte usate ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}",
        f"Feature rimosse ({len(REMOVED_FEATURES)}): "
        f"{', '.join(REMOVED_FEATURES.keys())}",
        "Motivo rimozione feature:",
    ]

    for feature, reason in REMOVED_FEATURES.items():
        lines.append(f"- {feature}: {reason}")

    lines.extend(
        [
            "",
            f"Feature escluse per evitare leakage: {', '.join(EXCLUDED_FEATURE_COLS)}",
            "Modello usato: KNeighborsClassifier",
            f"n_neighbors = {K_NEIGHBORS}",
            f"weights = {KNN_WEIGHTS}",
            "Metodo di validazione: Leave-One-Subject-Out",
            "Scaling: StandardScaler fittato solo sul training fold.",
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
            "## Confronto con baseline EDA completa",
            f"Baseline completa: accuracy={BASELINE_AGGREGATE_ACCURACY:.4f} "
            f"({format_percent(BASELINE_AGGREGATE_ACCURACY)}), "
            f"precision={BASELINE_AGGREGATE_PRECISION:.4f} "
            f"({format_percent(BASELINE_AGGREGATE_PRECISION)}), "
            f"recall={BASELINE_AGGREGATE_RECALL:.4f} "
            f"({format_percent(BASELINE_AGGREGATE_RECALL)}), "
            f"F1={BASELINE_AGGREGATE_F1:.4f} "
            f"({format_percent(BASELINE_AGGREGATE_F1)})",
            f"Delta accuracy ridotta-completa: "
            f"{summary_values['delta_accuracy']:+.4f} "
            f"({format_percent(summary_values['delta_accuracy'])})",
            f"Delta precision ridotta-completa: "
            f"{summary_values['delta_precision']:+.4f} "
            f"({format_percent(summary_values['delta_precision'])})",
            f"Delta recall ridotta-completa: "
            f"{summary_values['delta_recall']:+.4f} "
            f"({format_percent(summary_values['delta_recall'])})",
            f"Delta F1 ridotta-completa: {summary_values['delta_f1']:+.4f} "
            f"({format_percent(summary_values['delta_f1'])})",
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
            "## Soggetti con performance molto piu' bassa",
            f"Criterio: F1 < media fold F1 - {LOW_PERFORMANCE_MARGIN:.2f} "
            f"({summary_values['low_f1_threshold']:.4f}).",
        ]
    )

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

    if summary_values["delta_f1"] >= 0:
        lines.append(
            "La rimozione delle feature ridondanti non peggiora il F1 aggregato; "
            "la versione ridotta e' quindi preferibile come baseline piu' compatta."
        )
    else:
        lines.append(
            "La rimozione delle feature ridondanti riduce leggermente il F1 aggregato; "
            "la versione completa resta piu' forte, mentre quella ridotta e' piu' "
            "semplice e meno duplicata."
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
    section_title = "## Test EDA feature ridotte"
    section = f"""{section_title}

- Script creato: `src/train_eda_knn_loso_reduced.py`
- Dataset usato: `data_features/eda_features_all_60s.csv`
- Feature complete: `{len(FULL_FEATURE_COLS)}`
- Feature ridotte usate: `{len(FEATURE_COLS)}`
- Feature rimosse per ridondanza: `{", ".join(REMOVED_FEATURES.keys())}`
- Validazione: Leave-One-Subject-Out per soggetto, con `StandardScaler` fittato solo sul training fold.
- Modello: `KNeighborsClassifier(n_neighbors={K_NEIGHBORS}, weights="{KNN_WEIGHTS}")`
- Risultati feature ridotte:
  - Accuracy aggregata: {summary_values["aggregate_accuracy"]:.4f} ({format_percent(summary_values["aggregate_accuracy"])})
  - Precision aggregata: {summary_values["aggregate_precision"]:.4f} ({format_percent(summary_values["aggregate_precision"])})
  - Recall aggregata: {summary_values["aggregate_recall"]:.4f} ({format_percent(summary_values["aggregate_recall"])})
  - F1 aggregato: {summary_values["aggregate_f1"]:.4f} ({format_percent(summary_values["aggregate_f1"])})
  - Confusion matrix aggregata: TN={summary_values["tn"]}, FP={summary_values["fp"]}, FN={summary_values["fn"]}, TP={summary_values["tp"]}
- Confronto con baseline EDA completa:
  - Delta accuracy: {summary_values["delta_accuracy"]:+.4f} ({format_percent(summary_values["delta_accuracy"])})
  - Delta precision: {summary_values["delta_precision"]:+.4f} ({format_percent(summary_values["delta_precision"])})
  - Delta recall: {summary_values["delta_recall"]:+.4f} ({format_percent(summary_values["delta_recall"])})
  - Delta F1: {summary_values["delta_f1"]:+.4f} ({format_percent(summary_values["delta_f1"])})
- File generati:
  - `results/eda_knn_loso_reduced_per_subject.csv`
  - `results/eda_knn_reduced_summary.txt`
- Soggetti problematici secondo soglia F1: {low_subjects_text}
"""
    replace_or_append_progress_section(section_title, section)


def save_results(df, per_subject_df, y_true, y_pred):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_subject_df.to_csv(PER_SUBJECT_FILE, index=False)

    summary_values = calculate_summary_values(df, per_subject_df, y_true, y_pred)
    SUMMARY_FILE.write_text(
        build_summary(df, per_subject_df, y_true, y_pred, summary_values),
        encoding="utf-8",
    )
    update_progress(summary_values)
    return summary_values


def main():
    df = load_dataset()
    validate_dataset(df)
    per_subject_df, y_true, y_pred = run_loso_knn(df)
    summary_values = save_results(df, per_subject_df, y_true, y_pred)

    print("\nRisultati salvati:")
    print(f"- {PER_SUBJECT_FILE}")
    print(f"- {SUMMARY_FILE}")
    print(f"- {PROGRESS_FILE}")
    print("\nMetriche aggregate feature ridotte:")
    print(
        f"Accuracy={summary_values['aggregate_accuracy']:.4f}, "
        f"Precision={summary_values['aggregate_precision']:.4f}, "
        f"Recall={summary_values['aggregate_recall']:.4f}, "
        f"F1={summary_values['aggregate_f1']:.4f}"
    )
    print(
        "Delta vs baseline completa: "
        f"Accuracy={summary_values['delta_accuracy']:+.4f}, "
        f"Precision={summary_values['delta_precision']:+.4f}, "
        f"Recall={summary_values['delta_recall']:+.4f}, "
        f"F1={summary_values['delta_f1']:+.4f}"
    )


if __name__ == "__main__":
    main()
