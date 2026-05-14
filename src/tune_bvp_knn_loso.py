from pathlib import Path

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


DATASET_FILE = Path("data_features") / "bvp_features_all_60s.csv"
RESULTS_DIR = Path("results")
TUNING_RESULTS_FILE = RESULTS_DIR / "bvp_knn_tuning_results.csv"
TUNING_SUMMARY_FILE = RESULTS_DIR / "bvp_knn_tuning_summary.txt"
PROGRESS_FILE = Path("PROGRESS.md")

FEATURE_COLS = [
    "bvp_mean",
    "bvp_std",
    "bvp_min",
    "bvp_max",
    "bvp_range",
    "bvp_median",
    "bvp_iqr",
    "bvp_rms",
    "bvp_energy",
    "bvp_skew",
    "bvp_kurtosis",
]

TARGET_COL = "label"
GROUP_COL = "subject"
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]
WEIGHTS_VALUES = ["uniform", "distance"]
CURRENT_CONFIG = {"k": 9, "weights": "distance"}


def format_percent(value):
    return f"{value * 100:.2f}%"


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_FILE}")

    return pd.read_csv(DATASET_FILE)


def validate_dataset(df):
    required_columns = FEATURE_COLS + [TARGET_COL, GROUP_COL]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonne richieste mancanti: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset vuoto.")

    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 classi in y.")

    if df[GROUP_COL].nunique() < 2:
        raise ValueError("Servono almeno 2 soggetti per LOSO.")

    feature_nan_counts = df[FEATURE_COLS].isna().sum()
    if feature_nan_counts.any():
        raise ValueError(
            "NaN nelle feature: "
            f"{feature_nan_counts[feature_nan_counts > 0].to_dict()}"
        )


def evaluate_knn_config(df, k, weights):
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

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=k, weights=weights)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        fold_rows.append(
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, pos_label=1, zero_division=0
                ),
                "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            }
        )

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    fold_df = pd.DataFrame(fold_rows)
    y_true = pd.Series(all_true)
    y_pred = pd.Series(all_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    predicted_classes = [int(label) for label in sorted(y_pred.unique())]

    return {
        "k": k,
        "weights": weights,
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_classes": ",".join(str(label) for label in predicted_classes),
        "predicts_single_class": len(predicted_classes) == 1,
    }


def run_tuning(df):
    rows = []

    for k in K_VALUES:
        for weights in WEIGHTS_VALUES:
            print(f"Tuning kNN LOSO: k={k}, weights={weights}")
            rows.append(evaluate_knn_config(df, k, weights))

    results_df = pd.DataFrame(rows)
    return results_df.sort_values(
        ["aggregate_f1", "aggregate_accuracy"], ascending=[False, False]
    ).reset_index(drop=True)


def build_summary(df, results_df):
    best = results_df.iloc[0]
    current_rows = results_df[
        (results_df["k"] == CURRENT_CONFIG["k"])
        & (results_df["weights"] == CURRENT_CONFIG["weights"])
    ]
    current = current_rows.iloc[0]
    f1_improvement = best["aggregate_f1"] - current["aggregate_f1"]
    accuracy_improvement = best["aggregate_accuracy"] - current["aggregate_accuracy"]

    lines = [
        "# BVP wrist + kNN LOSO tuning summary",
        "",
        f"Dataset usato: {DATASET_FILE}",
        f"Feature usate: {', '.join(FEATURE_COLS)}",
        f"k_values: {K_VALUES}",
        f"weights_values: {WEIGHTS_VALUES}",
        "Metodo di validazione: Leave-One-Subject-Out",
        "Scaling: StandardScaler fittato solo sul training fold.",
        f"Numero soggetti: {df[GROUP_COL].nunique()}",
        f"Numero campioni: {len(df)}",
        "",
        "## Migliore configurazione per F1 aggregato",
        f"k={int(best['k'])}, weights={best['weights']}",
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
        "## Confronto con configurazione corrente",
        f"Corrente: k={CURRENT_CONFIG['k']}, weights={CURRENT_CONFIG['weights']}",
        f"F1 corrente: {current['aggregate_f1']:.4f} "
        f"({format_percent(current['aggregate_f1'])})",
        f"Accuracy corrente: {current['aggregate_accuracy']:.4f} "
        f"({format_percent(current['aggregate_accuracy'])})",
        f"Delta F1 migliore-corrente: {f1_improvement:+.4f} "
        f"({format_percent(f1_improvement)})",
        f"Delta accuracy migliore-corrente: {accuracy_improvement:+.4f} "
        f"({format_percent(accuracy_improvement)})",
        "",
        "## Top configurazioni",
        results_df.head(10).to_string(index=False),
        "",
        "## Interpretazione",
    ]

    if best["aggregate_f1"] < 0.60:
        lines.append(
            "Il tuning resta vicino all'area 55-60% F1: il prossimo salto "
            "probabile e' aggiungere feature fisiologiche da peak detection/HRV."
        )
    else:
        lines.append(
            "Il tuning migliora in modo visibile il F1: conviene usare questa "
            "configurazione come nuova baseline prima di passare a feature HRV."
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
    section_title = "## Tuning leggero kNN BVP"
    section = f"""{section_title}

- Script creato: `src/tune_bvp_knn_loso.py`
- Output:
  - `results/bvp_knn_tuning_results.csv`
  - `results/bvp_knn_tuning_summary.txt`
- Esperimenti: `k={K_VALUES}` con `weights={WEIGHTS_VALUES}`
- Validazione: Leave-One-Subject-Out, con `StandardScaler` fittato solo sul training fold.
- Migliore configurazione per F1 aggregato: `k={int(best["k"])}`, `weights="{best["weights"]}"`
- Risultati migliori:
  - Accuracy aggregata: {best["aggregate_accuracy"]:.4f} ({format_percent(best["aggregate_accuracy"])})
  - Precision aggregata: {best["aggregate_precision"]:.4f} ({format_percent(best["aggregate_precision"])})
  - Recall aggregata: {best["aggregate_recall"]:.4f} ({format_percent(best["aggregate_recall"])})
  - F1 aggregato: {best["aggregate_f1"]:.4f} ({format_percent(best["aggregate_f1"])})
  - Confusion matrix: TN={int(best["tn"])}, FP={int(best["fp"])}, FN={int(best["fn"])}, TP={int(best["tp"])}
"""
    replace_or_append_progress_section(section_title, section)


def save_results(df, results_df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TUNING_RESULTS_FILE, index=False)
    TUNING_SUMMARY_FILE.write_text(build_summary(df, results_df), encoding="utf-8")
    update_progress(results_df)


def main():
    df = load_dataset()
    validate_dataset(df)
    results_df = run_tuning(df)
    save_results(df, results_df)

    best = results_df.iloc[0]
    print("\nTuning completato.")
    print(f"File risultati: {TUNING_RESULTS_FILE}")
    print(f"File summary: {TUNING_SUMMARY_FILE}")
    print(
        "Migliore configurazione: "
        f"k={int(best['k'])}, weights={best['weights']}, "
        f"F1={best['aggregate_f1']:.4f}, "
        f"Accuracy={best['aggregate_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
