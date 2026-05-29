from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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
RESULTS_FILE = RESULTS_DIR / "eda_knn_baseline_norm_results.csv"
BEST_PER_SUBJECT_FILE = RESULTS_DIR / "eda_knn_baseline_norm_best_per_subject.csv"
SUMMARY_FILE = RESULTS_DIR / "eda_knn_baseline_norm_summary.txt"
PROGRESS_FILE = Path("PROGRESS.md")

TARGET_COL = "label"
GROUP_COL = "subject"
BASELINE_LABEL = 1
K_NEIGHBORS = 9
KNN_WEIGHTS = "distance"
EPSILON = 1e-9

REDUCED_FEATURE_COLS = [
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

BASELINE_NORMALIZED_FEATURES = [
    "eda_mean",
    "eda_std",
    "eda_range",
    "eda_tonic_std",
    "eda_phasic_std",
    "eda_phasic_auc",
    "eda_scr_count",
    "eda_scr_amplitude_std",
    "eda_scr_amplitude_max",
    "eda_scr_prominence_mean",
]

REFERENCE_METRICS = {
    "eda_full_28": {
        "accuracy": 0.8559,
        "precision": 0.7720,
        "recall": 0.7383,
        "f1": 0.7548,
    },
    "eda_reduced_16": {
        "accuracy": 0.8597,
        "precision": 0.7785,
        "recall": 0.7445,
        "f1": 0.7611,
    },
}


def format_percent(value):
    return f"{value * 100:.2f}%"


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_FILE}")
    return pd.read_csv(DATASET_FILE)


def validate_dataset(df):
    required_columns = (
        REDUCED_FEATURE_COLS
        + [TARGET_COL, GROUP_COL, "original_label", "label_purity"]
    )
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonne richieste mancanti: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset vuoto.")
    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 classi in y.")
    if df[GROUP_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 soggetti per LOSO.")
    if not np.isfinite(df[REDUCED_FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("Feature EDA non finite.")


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

    return df


def get_feature_sets():
    delta_features = [
        f"{feature}_baseline_delta" for feature in BASELINE_NORMALIZED_FEATURES
    ]
    zscore_features = [
        f"{feature}_baseline_zscore" for feature in BASELINE_NORMALIZED_FEATURES
    ]
    return {
        "reduced_absolute_16": REDUCED_FEATURE_COLS,
        "baseline_delta_10": delta_features,
        "baseline_zscore_10": zscore_features,
        "reduced_plus_delta_26": REDUCED_FEATURE_COLS + delta_features,
        "reduced_plus_zscore_26": REDUCED_FEATURE_COLS + zscore_features,
        "reduced_plus_delta_zscore_36": REDUCED_FEATURE_COLS
        + delta_features
        + zscore_features,
    }


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

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        left_out_subject = groups.iloc[test_idx].iloc[0]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights=KNN_WEIGHTS)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

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
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    y_true = pd.Series(all_true)
    y_pred = pd.Series(all_pred)
    tn, fp, fn, tp = get_binary_confusion_values(y_true, y_pred)
    predicted_classes = ",".join(str(int(label)) for label in sorted(y_pred.unique()))

    return {
        "summary": {
            "feature_set": feature_set_name,
            "n_features": len(feature_cols),
            "features": ",".join(feature_cols),
            "mean_accuracy": np.mean([row["accuracy"] for row in fold_rows]),
            "mean_precision": np.mean([row["precision"] for row in fold_rows]),
            "mean_recall": np.mean([row["recall"] for row in fold_rows]),
            "mean_f1": np.mean([row["f1"] for row in fold_rows]),
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
        },
        "fold_rows": fold_rows,
    }


def run_tests(df):
    rows = []
    fold_rows = []

    for feature_set_name, feature_cols in get_feature_sets().items():
        print(
            f"Test EDA baseline norm: {feature_set_name} "
            f"({len(feature_cols)} feature)"
        )
        result = evaluate_feature_set(df, feature_set_name, feature_cols)
        rows.append(result["summary"])
        fold_rows.extend(result["fold_rows"])

    results_df = pd.DataFrame(rows).sort_values(
        ["aggregate_f1", "aggregate_accuracy"], ascending=[False, False]
    )
    fold_df = pd.DataFrame(fold_rows)
    return results_df.reset_index(drop=True), fold_df


def build_summary(df, results_df):
    best = results_df.iloc[0]
    reduced_row = results_df[
        results_df["feature_set"] == "reduced_absolute_16"
    ].iloc[0]

    lines = [
        "# EDA baseline-normalized feature test",
        "",
        f"Dataset usato: {DATASET_FILE}",
        "Validazione: Leave-One-Subject-Out.",
        "Scaling: StandardScaler fittato solo sul training fold.",
        f"Modello: KNeighborsClassifier(n_neighbors={K_NEIGHBORS}, "
        f"weights='{KNN_WEIGHTS}')",
        "",
        "## Assunzione metodologica",
        "Le feature normalizzate rispetto alla baseline usano solo finestre con "
        "`original_label=1` dello stesso soggetto per stimare media e deviazione "
        "standard personali. Questo equivale ad assumere una fase di calibrazione "
        "baseline disponibile anche per il soggetto test; non usa finestre stress "
        "per costruire il riferimento.",
        "",
        f"Numero soggetti: {df[GROUP_COL].nunique()}",
        f"Numero campioni: {len(df)}",
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
        "",
        "## Confronto con riferimenti precedenti",
        f"EDA completa 28 feature: F1={REFERENCE_METRICS['eda_full_28']['f1']:.4f} "
        f"({format_percent(REFERENCE_METRICS['eda_full_28']['f1'])}), "
        f"Accuracy={REFERENCE_METRICS['eda_full_28']['accuracy']:.4f} "
        f"({format_percent(REFERENCE_METRICS['eda_full_28']['accuracy'])})",
        f"EDA ridotta 16 feature precedente: "
        f"F1={REFERENCE_METRICS['eda_reduced_16']['f1']:.4f} "
        f"({format_percent(REFERENCE_METRICS['eda_reduced_16']['f1'])}), "
        f"Accuracy={REFERENCE_METRICS['eda_reduced_16']['accuracy']:.4f} "
        f"({format_percent(REFERENCE_METRICS['eda_reduced_16']['accuracy'])})",
        f"Delta F1 migliore - ridotta precedente: "
        f"{best['aggregate_f1'] - REFERENCE_METRICS['eda_reduced_16']['f1']:+.4f} "
        f"({format_percent(best['aggregate_f1'] - REFERENCE_METRICS['eda_reduced_16']['f1'])})",
        f"Delta accuracy migliore - ridotta precedente: "
        f"{best['aggregate_accuracy'] - REFERENCE_METRICS['eda_reduced_16']['accuracy']:+.4f} "
        f"({format_percent(best['aggregate_accuracy'] - REFERENCE_METRICS['eda_reduced_16']['accuracy'])})",
        "",
        "## Controllo replica feature ridotte",
        f"Feature set reduced_absolute_16 in questo script: "
        f"F1={reduced_row['aggregate_f1']:.4f} "
        f"({format_percent(reduced_row['aggregate_f1'])}), "
        f"Accuracy={reduced_row['aggregate_accuracy']:.4f} "
        f"({format_percent(reduced_row['aggregate_accuracy'])})",
        "",
        "## Tutti i risultati",
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
            ]
        ].to_string(index=False),
        "",
        "## Interpretazione",
    ]

    if best["aggregate_f1"] > REFERENCE_METRICS["eda_reduced_16"]["f1"]:
        lines.append(
            "La normalizzazione rispetto alla baseline personale migliora il F1 "
            "rispetto alla versione ridotta precedente. Il beneficio indica che "
            "rimuovere parte della variabilita' individuale aiuta il kNN in LOSO."
        )
    else:
        lines.append(
            "La normalizzazione rispetto alla baseline personale non migliora il F1 "
            "rispetto alla versione ridotta precedente. In questa configurazione "
            "conviene mantenere le feature ridotte assolute."
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
    section_title = "## Test EDA normalizzazione baseline soggetto"
    section = f"""{section_title}

- Script creato: `src/train_eda_knn_loso_baseline_norm.py`
- Dataset usato: `data_features/eda_features_all_60s.csv`
- Assunzione: baseline personale disponibile per ogni soggetto, usando solo finestre con `original_label=1`.
- Feature set testati: `reduced_absolute_16`, `baseline_delta_10`, `baseline_zscore_10`, `reduced_plus_delta_26`, `reduced_plus_zscore_26`, `reduced_plus_delta_zscore_36`
- Migliore feature set: `{best["feature_set"]}` con `{int(best["n_features"])}` feature.
- Risultati migliori:
  - Accuracy aggregata: {best["aggregate_accuracy"]:.4f} ({format_percent(best["aggregate_accuracy"])})
  - Precision aggregata: {best["aggregate_precision"]:.4f} ({format_percent(best["aggregate_precision"])})
  - Recall aggregata: {best["aggregate_recall"]:.4f} ({format_percent(best["aggregate_recall"])})
  - F1 aggregato: {best["aggregate_f1"]:.4f} ({format_percent(best["aggregate_f1"])})
  - Confusion matrix: TN={int(best["tn"])}, FP={int(best["fp"])}, FN={int(best["fn"])}, TP={int(best["tp"])}
- Delta F1 vs EDA ridotta 16 feature precedente: {best["aggregate_f1"] - REFERENCE_METRICS["eda_reduced_16"]["f1"]:+.4f} ({format_percent(best["aggregate_f1"] - REFERENCE_METRICS["eda_reduced_16"]["f1"])})
- File generati:
  - `results/eda_knn_baseline_norm_results.csv`
  - `results/eda_knn_baseline_norm_best_per_subject.csv`
  - `results/eda_knn_baseline_norm_summary.txt`
"""
    replace_or_append_progress_section(section_title, section)


def save_results(df, results_df, fold_df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_FILE, index=False)
    best_feature_set = results_df.iloc[0]["feature_set"]
    fold_df[fold_df["feature_set"] == best_feature_set].to_csv(
        BEST_PER_SUBJECT_FILE, index=False
    )
    SUMMARY_FILE.write_text(build_summary(df, results_df), encoding="utf-8")
    update_progress(results_df)


def main():
    df = load_dataset()
    validate_dataset(df)
    df = add_baseline_normalized_features(df)
    results_df, fold_df = run_tests(df)
    save_results(df, results_df, fold_df)

    best = results_df.iloc[0]
    print("\nTest normalizzazione baseline completato.")
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
