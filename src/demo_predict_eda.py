import argparse
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


TRAIN_FILE = Path("data_features") / "eda_features_train_s2_s16_60s.csv"
TEST_FILE = Path("data_features") / "eda_features_test_s17_60s.csv"
RESULTS_DIR = Path("results")
DEMO_OUTPUT_FILE = RESULTS_DIR / "demo_predictions_eda_s17.csv"

FEATURE_COLS = [
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

TARGET_COL = "label"
K_NEIGHBORS = 9
KNN_WEIGHTS = "distance"
LABEL_NAMES = {
    0: "non-stress",
    1: "stress",
}


def format_percent(value):
    return f"{value * 100:.2f}%"


def label_name(label):
    return LABEL_NAMES.get(int(label), f"classe {label}")


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    return pd.read_csv(path)


def validate_dataset(df, dataset_name):
    required_columns = FEATURE_COLS + [
        TARGET_COL,
        "subject",
        "start_sec",
        "end_sec",
        "original_label",
        "label_purity",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"{dataset_name}: colonne mancanti: {missing_columns}")
    if df.empty:
        raise ValueError(f"{dataset_name}: dataset vuoto.")
    if not np.isfinite(df[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError(f"{dataset_name}: NaN o valori infiniti nelle feature.")


def train_model(train_df):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights=KNN_WEIGHTS)
    model.fit(X_train_scaled, y_train)
    return scaler, model


def predict_test_set(test_df, scaler, model):
    X_test_scaled = scaler.transform(test_df[FEATURE_COLS])
    predictions_df = test_df[
        ["subject", "start_sec", "end_sec", "original_label", "label", "label_purity"]
    ].copy()
    predictions_df["predicted_label"] = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)
    probability_index = {
        int(class_label): index for index, class_label in enumerate(model.classes_)
    }
    stress_probability_index = probability_index.get(1)
    predictions_df["stress_probability"] = (
        probabilities[:, stress_probability_index]
        if stress_probability_index is not None
        else 0.0
    )
    predictions_df["true_label_name"] = predictions_df["label"].map(label_name)
    predictions_df["predicted_label_name"] = predictions_df["predicted_label"].map(
        label_name
    )
    predictions_df["correct"] = (
        predictions_df["label"] == predictions_df["predicted_label"]
    )
    return predictions_df


def select_demo_rows(predictions_df, n_rows, random_state):
    if n_rows >= len(predictions_df):
        return predictions_df.copy()

    label_values = sorted(predictions_df[TARGET_COL].unique())
    rows_per_label = max(n_rows // len(label_values), 1)
    sampled_parts = []

    for offset, label in enumerate(label_values):
        label_rows = predictions_df[predictions_df[TARGET_COL] == label]
        sampled_parts.append(
            label_rows.sample(
                n=min(rows_per_label, len(label_rows)),
                random_state=random_state + offset,
            )
        )

    sampled_df = pd.concat(sampled_parts)
    remaining_rows = n_rows - len(sampled_df)
    if remaining_rows > 0:
        candidates = predictions_df.drop(index=sampled_df.index)
        sampled_df = pd.concat(
            [
                sampled_df,
                candidates.sample(
                    n=min(remaining_rows, len(candidates)),
                    random_state=random_state + 100,
                ),
            ]
        )

    return sampled_df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)


def print_metrics(predictions_df):
    y_true = predictions_df["label"]
    y_pred = predictions_df["predicted_label"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print("Metriche su tutto il test set S17")
    print(
        f"- Accuracy: {accuracy_score(y_true, y_pred):.4f} "
        f"({format_percent(accuracy_score(y_true, y_pred))})"
    )
    print(
        f"- Precision stress: {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f} "
        f"({format_percent(precision_score(y_true, y_pred, pos_label=1, zero_division=0))})"
    )
    print(
        f"- Recall stress: {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f} "
        f"({format_percent(recall_score(y_true, y_pred, pos_label=1, zero_division=0))})"
    )
    print(
        f"- F1 stress: {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f} "
        f"({format_percent(f1_score(y_true, y_pred, pos_label=1, zero_division=0))})"
    )
    print(f"- Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def print_demo_rows(demo_df):
    print(f"\nEsempi di predizione su {len(demo_df)} finestre di S17")

    for _, row in demo_df.iterrows():
        result = "GIUSTO" if row["correct"] else "SBAGLIATO"
        print(
            f"- {row['subject']} {int(row['start_sec'])}-{int(row['end_sec'])}s | "
            f"vero: {row['true_label_name']} | "
            f"predetto: {row['predicted_label_name']} | "
            f"P(stress): {row['stress_probability']:.2f} | "
            f"{result}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Demo kNN EDA: allena su S2-S16 e mostra predizioni sul test set S17."
        )
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=15,
        help="Numero di finestre di S17 da mostrare nella demo.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed per scegliere esempi riproducibili.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rows < 1:
        raise ValueError("--rows deve essere almeno 1.")

    train_df = load_csv(TRAIN_FILE)
    test_df = load_csv(TEST_FILE)
    validate_dataset(train_df, "Train S2-S16")
    validate_dataset(test_df, "Test S17")

    scaler, model = train_model(train_df)
    predictions_df = predict_test_set(test_df, scaler, model)
    demo_df = select_demo_rows(predictions_df, args.rows, args.random_state)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    demo_df.to_csv(DEMO_OUTPUT_FILE, index=False)

    print("Demo predizione stress da EDA")
    print(f"Train: {TRAIN_FILE} ({len(train_df)} finestre)")
    print(f"Test: {TEST_FILE} ({len(test_df)} finestre)")
    print(f"Modello: kNN k={K_NEIGHBORS}, weights={KNN_WEIGHTS}")
    print()
    print_metrics(predictions_df)
    print_demo_rows(demo_df)
    print(f"\nPredizioni demo salvate in: {DEMO_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
