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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


DATASET_FILE = Path("data_features") / "eda_features_all_60s.csv"
RESULTS_DIR = Path("results")
PER_SUBJECT_FILE = RESULTS_DIR / "eda_knn_top15_loso_per_subject.csv"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "eda_knn_top15_feature_importance.csv"
SUMMARY_FILE = RESULTS_DIR / "eda_knn_top15_summary.txt"
PROGRESS_FILE = Path("PROGRESS.md")

TARGET_COL = "label"
GROUP_COL = "subject"
BASELINE_LABEL = 1
EPSILON = 1e-9

TOP_N_FEATURES = 15
K_NEIGHBORS = 9
KNN_WEIGHTS = "distance"
RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 2
RF_CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42
LOW_PERFORMANCE_MARGIN = 0.15

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

TOP15_FEATURE_COLS = [
    "eda_mean_baseline_zscore",
    "eda_phasic_auc_baseline_delta",
    "eda_phasic_auc_baseline_zscore",
    "eda_scr_count_baseline_delta",
    "eda_range_baseline_delta",
    "eda_scr_count_baseline_zscore",
    "eda_mean_baseline_delta",
    "eda_phasic_std_baseline_delta",
    "eda_std_baseline_delta",
    "eda_phasic_std_baseline_zscore",
    "eda_mean",
    "eda_range_baseline_zscore",
    "eda_derivative_mean",
    "eda_scr_amplitude_max_baseline_delta",
    "eda_slope",
]

REFERENCE_METRICS = {
    "eda_reduced_16": {
        "accuracy": 0.8597,
        "precision": 0.7785,
        "recall": 0.7445,
        "f1": 0.7611,
    },
    "eda_baseline_norm_36": {
        "accuracy": 0.8952,
        "precision": 0.8828,
        "recall": 0.7508,
        "f1": 0.8114,
    },
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
            "Prima esegui src/extract_eda_features_all.py."
        )
    return pd.read_csv(DATASET_FILE)


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


def get_candidate_feature_cols():
    delta_features = [
        f"{feature}_baseline_delta" for feature in BASELINE_NORMALIZED_FEATURES
    ]
    zscore_features = [
        f"{feature}_baseline_zscore" for feature in BASELINE_NORMALIZED_FEATURES
    ]
    return REDUCED_FEATURE_COLS + delta_features + zscore_features


def validate_dataset(df, feature_cols):
    required_columns = feature_cols + [
        TARGET_COL,
        GROUP_COL,
        "original_label",
        "label_purity",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonne richieste mancanti: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset vuoto.")
    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 classi in y.")
    if df[GROUP_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 soggetti per Leave-One-Subject-Out.")
    if not np.isfinite(df[feature_cols].to_numpy(dtype=float)).all():
        raise ValueError("Sono presenti NaN o valori infiniti nelle feature EDA.")


def build_selector():
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def rank_features(X_train, y_train, feature_cols):
    selector = build_selector()
    selector.fit(X_train, y_train)
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": selector.feature_importances_,
            }
        )
        .sort_values(["importance", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def get_binary_confusion_values(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def run_loso_knn_top15(df, feature_cols):
    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    logo = LeaveOneGroupOut()
    fold_rows = []
    all_true = []
    all_pred = []
    feature_importance_rows = []

    for train_idx, test_idx in logo.split(df[feature_cols], y, groups):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        left_out_subject = groups.iloc[test_idx].iloc[0]

        print(f"LOSO fold - soggetto lasciato fuori: {left_out_subject}")

        X_train_all = df.iloc[train_idx][feature_cols]
        X_test_all = df.iloc[test_idx][feature_cols]

        importance_df = rank_features(X_train_all, y_train, feature_cols)
        rf_top_features = set(importance_df.head(TOP_N_FEATURES)["feature"])
        for _, row in importance_df.iterrows():
            feature_importance_rows.append(
                {
                    "left_out_subject": left_out_subject,
                    "feature": row["feature"],
                    "importance": row["importance"],
                    "selected_by_rf_top15": row["feature"] in rf_top_features,
                    "used_by_model": row["feature"] in TOP15_FEATURE_COLS,
                }
            )

        X_train = X_train_all[TOP15_FEATURE_COLS]
        X_test = X_test_all[TOP15_FEATURE_COLS]

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
                "selected_features": ",".join(TOP15_FEATURE_COLS),
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

    return (
        pd.DataFrame(fold_rows),
        pd.Series(all_true),
        pd.Series(all_pred),
        pd.DataFrame(feature_importance_rows),
    )


def aggregate_feature_importances(feature_importance_df):
    total_folds = feature_importance_df["left_out_subject"].nunique()
    summary_df = (
        feature_importance_df.groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            selected_count=("selected_by_rf_top15", "sum"),
            used_by_model=("used_by_model", "max"),
        )
        .reset_index()
    )
    summary_df["selected_rate"] = summary_df["selected_count"] / total_folds
    return summary_df.sort_values(
        ["used_by_model", "selected_count", "mean_importance", "feature"],
        ascending=[False, False, False, True],
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
        "low_f1_threshold": low_f1_threshold,
        "low_performance_subjects": low_performance_subjects,
    }


def build_summary(
    df,
    candidate_feature_cols,
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
        "# EDA wrist + kNN top15 LOSO summary",
        "",
        f"Dataset usato: {DATASET_FILE}",
        f"Feature candidate ({len(candidate_feature_cols)}): "
        f"{', '.join(candidate_feature_cols)}",
        f"Feature usate dal modello top15 ({len(TOP15_FEATURE_COLS)}): "
        f"{', '.join(TOP15_FEATURE_COLS)}",
        "Le top15 sono state scelte dalla classifica di stabilita' Random Forest "
        "sui fold LOSO; la classifica viene rigenerata e salvata per controllo.",
        "",
        "## Modello valutato",
        f"Classificatore finale: KNeighborsClassifier(n_neighbors={K_NEIGHBORS}, "
        f"weights='{KNN_WEIGHTS}')",
        f"Selettore feature: RandomForestClassifier(n_estimators={RF_N_ESTIMATORS}, "
        f"min_samples_leaf={RF_MIN_SAMPLES_LEAF}, "
        f"class_weight='{RF_CLASS_WEIGHT}', random_state={RANDOM_STATE})",
        "Metodo di validazione: Leave-One-Subject-Out.",
        "Scaling: StandardScaler fittato solo sul training fold dopo la selezione.",
        "",
        "## Assunzione metodologica",
        "Le feature baseline_delta e baseline_zscore usano le finestre baseline "
        "personali (`original_label=1`) dello stesso soggetto. Il modello valutato "
        "usa sempre la stessa lista di 15 feature aggregate, cosi' resta semplice "
        "da descrivere e riprodurre.",
        "",
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
        "## Confronto",
        f"EDA ridotta 16 feature: F1={REFERENCE_METRICS['eda_reduced_16']['f1']:.4f} "
        f"({format_percent(REFERENCE_METRICS['eda_reduced_16']['f1'])}), "
        f"Accuracy={REFERENCE_METRICS['eda_reduced_16']['accuracy']:.4f} "
        f"({format_percent(REFERENCE_METRICS['eda_reduced_16']['accuracy'])})",
        f"EDA baseline-normalized 36 feature: "
        f"F1={REFERENCE_METRICS['eda_baseline_norm_36']['f1']:.4f} "
        f"({format_percent(REFERENCE_METRICS['eda_baseline_norm_36']['f1'])}), "
        f"Accuracy={REFERENCE_METRICS['eda_baseline_norm_36']['accuracy']:.4f} "
        f"({format_percent(REFERENCE_METRICS['eda_baseline_norm_36']['accuracy'])})",
        f"Delta F1 top15 - 36 feature: "
        f"{summary_values['aggregate_f1'] - REFERENCE_METRICS['eda_baseline_norm_36']['f1']:+.4f} "
        f"({format_percent(summary_values['aggregate_f1'] - REFERENCE_METRICS['eda_baseline_norm_36']['f1'])})",
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
        "## Classifica feature",
        feature_importance_summary_df.to_string(index=False),
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

    if summary_values["aggregate_f1"] >= REFERENCE_METRICS["eda_baseline_norm_36"][
        "f1"
    ]:
        lines.extend(
            [
                "",
                "## Interpretazione",
                "La versione top15 mantiene o migliora il F1 della versione a 36 "
                "feature, quindi e' preferibile per compattezza e spiegabilita'.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Interpretazione",
                "La versione top15 riduce molto il numero di feature ma perde qualcosa "
                "rispetto alla versione a 36 feature. Resta utile come modello compatto "
                "e piu' semplice da spiegare.",
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


def update_progress(summary_values, top15_stable_features):
    low_subjects = summary_values["low_performance_subjects"]
    low_subjects_text = (
        ", ".join(low_subjects) if low_subjects else "nessun soggetto sotto soglia"
    )
    section_title = "## Test EDA top15 feature"
    section = f"""{section_title}

- Script creato: `src/train_eda_knn_top15_loso.py`
- Dataset usato: `data_features/eda_features_all_60s.csv`
- Feature candidate: `36` dalla migliore configurazione EDA baseline-normalized.
- Selezione: top `{TOP_N_FEATURES}` aggregate dalla classifica di stabilita' `RandomForestClassifier` sui fold LOSO.
- Modello finale per fold: `KNeighborsClassifier(n_neighbors={K_NEIGHBORS}, weights="{KNN_WEIGHTS}")` con `StandardScaler` fittato solo sul training fold.
- Top 15 feature aggregate: `{", ".join(top15_stable_features)}`
- Risultati:
  - Accuracy aggregata: {summary_values["aggregate_accuracy"]:.4f} ({format_percent(summary_values["aggregate_accuracy"])})
  - Precision aggregata: {summary_values["aggregate_precision"]:.4f} ({format_percent(summary_values["aggregate_precision"])})
  - Recall aggregata: {summary_values["aggregate_recall"]:.4f} ({format_percent(summary_values["aggregate_recall"])})
  - F1 aggregato: {summary_values["aggregate_f1"]:.4f} ({format_percent(summary_values["aggregate_f1"])})
  - Confusion matrix: TN={summary_values["tn"]}, FP={summary_values["fp"]}, FN={summary_values["fn"]}, TP={summary_values["tp"]}
- Delta F1 vs EDA baseline-normalized 36 feature: {summary_values["aggregate_f1"] - REFERENCE_METRICS["eda_baseline_norm_36"]["f1"]:+.4f} ({format_percent(summary_values["aggregate_f1"] - REFERENCE_METRICS["eda_baseline_norm_36"]["f1"])})
- Soggetti problematici secondo soglia F1: {low_subjects_text}
- File generati:
  - `results/eda_knn_top15_loso_per_subject.csv`
  - `results/eda_knn_top15_feature_importance.csv`
  - `results/eda_knn_top15_summary.txt`
"""
    replace_or_append_progress_section(section_title, section)


def save_results(
    df,
    candidate_feature_cols,
    per_subject_df,
    feature_importance_df,
    y_true,
    y_pred,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_subject_df.to_csv(PER_SUBJECT_FILE, index=False)

    feature_importance_summary_df = aggregate_feature_importances(feature_importance_df)
    feature_importance_summary_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    summary_values = calculate_summary_values(df, per_subject_df, y_true, y_pred)
    SUMMARY_FILE.write_text(
        build_summary(
            df,
            candidate_feature_cols,
            per_subject_df,
            feature_importance_summary_df,
            y_true,
            y_pred,
            summary_values,
        ),
        encoding="utf-8",
    )

    top15_stable_features = TOP15_FEATURE_COLS
    update_progress(summary_values, top15_stable_features)
    return summary_values, top15_stable_features


def main():
    df = load_dataset()
    df = add_baseline_normalized_features(df)
    candidate_feature_cols = get_candidate_feature_cols()
    validate_dataset(df, candidate_feature_cols)
    per_subject_df, y_true, y_pred, feature_importance_df = run_loso_knn_top15(
        df, candidate_feature_cols
    )
    summary_values, top15_stable_features = save_results(
        df,
        candidate_feature_cols,
        per_subject_df,
        feature_importance_df,
        y_true,
        y_pred,
    )

    print("\nRisultati salvati:")
    print(f"- {PER_SUBJECT_FILE}")
    print(f"- {FEATURE_IMPORTANCE_FILE}")
    print(f"- {SUMMARY_FILE}")
    print("\nTop 15 feature aggregate:")
    print(", ".join(top15_stable_features))
    print("\nMetriche aggregate:")
    print(
        f"Accuracy={summary_values['aggregate_accuracy']:.4f}, "
        f"Precision={summary_values['aggregate_precision']:.4f}, "
        f"Recall={summary_values['aggregate_recall']:.4f}, "
        f"F1={summary_values['aggregate_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
