import argparse
from pathlib import Path

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


TRAIN_FILE = Path("data_features") / "bvp_features_train_s2_s16_60s.csv"
TEST_FILE = Path("data_features") / "bvp_features_test_s17_60s.csv"
RESULTS_DIR = Path("results")
DEMO_OUTPUT_FILE = RESULTS_DIR / "demo_predictions_s17.csv"

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
    required_columns = FEATURE_COLS + [TARGET_COL, "subject", "start_sec", "end_sec"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"{dataset_name}: colonne mancanti: {missing_columns}")

    if df.empty:
        raise ValueError(f"{dataset_name}: dataset vuoto.")

    feature_nan_counts = df[FEATURE_COLS].isna().sum()
    if feature_nan_counts.any():
        raise ValueError(
            f"{dataset_name}: NaN nelle feature: "
            f"{feature_nan_counts[feature_nan_counts > 0].to_dict()}"
        )


def train_model(train_df):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights=KNN_WEIGHTS)
    model.fit(X_train_scaled, y_train)

    return scaler, model


def predict_test_set(test_df, scaler, model):
    X_test = test_df[FEATURE_COLS]
    X_test_scaled = scaler.transform(X_test)

    predictions_df = test_df[
        ["subject", "start_sec", "end_sec", "original_label", "label", "label_purity"]
    ].copy()
    predictions_df["predicted_label"] = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)

    class_to_probability_index = {
        int(class_label): index for index, class_label in enumerate(model.classes_)
    }
    stress_probability_index = class_to_probability_index.get(1)

    if stress_probability_index is None:
        predictions_df["stress_probability"] = 0.0
    else:
        predictions_df["stress_probability"] = probabilities[:, stress_probability_index]

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
        sample_size = min(rows_per_label, len(label_rows))
        sampled_parts.append(
            label_rows.sample(n=sample_size, random_state=random_state + offset)
        )

    sampled_df = pd.concat(sampled_parts)
    remaining_rows = n_rows - len(sampled_df)

    if remaining_rows > 0:
        remaining_candidates = predictions_df.drop(index=sampled_df.index)
        if not remaining_candidates.empty:
            sampled_df = pd.concat(
                [
                    sampled_df,
                    remaining_candidates.sample(
                        n=min(remaining_rows, len(remaining_candidates)),
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
    print(f"- Accuracy: {accuracy_score(y_true, y_pred):.4f} ({format_percent(accuracy_score(y_true, y_pred))})")
    print(
        f"- Precision stress: {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f} "
        f"({format_percent(precision_score(y_true, y_pred, pos_label=1, zero_division=0))})"
    )
    print(
        f"- Recall stress: {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f} "
        f"({format_percent(recall_score(y_true, y_pred, pos_label=1, zero_division=0))})"
    )
    print(f"- F1 stress: {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f} ({format_percent(f1_score(y_true, y_pred, pos_label=1, zero_division=0))})")
    print(f"- Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def print_demo_rows(demo_df):
    print("\nEsempi di predizione su alcune finestre di S17")

    for _, row in demo_df.iterrows():
        result = "OK" if row["correct"] else "ERRORE"
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
            "Demo kNN BVP: allena su S2-S16 e mostra alcune predizioni sul test set S17."
        )
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=12,
        help="Numero di righe di S17 da mostrare nella demo.",
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

    train_df = load_csv(TRAIN_FILE)
    test_df = load_csv(TEST_FILE)
    validate_dataset(train_df, "Train S2-S16")
    validate_dataset(test_df, "Test S17")

    scaler, model = train_model(train_df)
    predictions_df = predict_test_set(test_df, scaler, model)
    demo_df = select_demo_rows(predictions_df, args.rows, args.random_state)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    demo_df.to_csv(DEMO_OUTPUT_FILE, index=False)

    print("Demo predizione stress da BVP")
    print(f"Train: {TRAIN_FILE} ({len(train_df)} finestre)")
    print(f"Test: {TEST_FILE} ({len(test_df)} finestre)")
    print(f"Modello: kNN k={K_NEIGHBORS}, weights={KNN_WEIGHTS}")
    print()

    print_metrics(predictions_df)
    print_demo_rows(demo_df)

    print(f"\nPredizioni demo salvate in: {DEMO_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
