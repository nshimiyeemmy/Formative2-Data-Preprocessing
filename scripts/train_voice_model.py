import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib


def infer_speaker_from_filename(fname: str) -> str:
    """
    Infer speaker name from file_name.
    Example: 'tracy_yes_1.wav' -> 'tracy'
             'eva_confirm2.m4a' -> 'eva'
    """
    stem = Path(fname).stem  # remove extension
    # take first chunk before '_' or '-' etc.
    parts = stem.replace("-", "_").split("_")
    return parts[0].lower() if parts else stem.lower()


def main():
    # ---------------------------------------------
    # 1. Locate project paths
    # ---------------------------------------------
    script_dir = Path(__file__).resolve().parent     # .../scripts
    project_root = script_dir.parent                 # project root

    features_path = project_root / "Audio_Processing" / "audio_features.csv"
    models_dir = project_root / "Audio_Processing" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Loading audio features from: {features_path}")

    if not features_path.exists():
        print(" audio_features.csv not found. Make sure it exists in 'Audio_Processing/'")
        return

    # ---------------------------------------------
    # 2. Load dataset
    # ---------------------------------------------
    df = pd.read_csv(features_path)

    # Try to detect an existing target column
    possible_targets = ["member_name", "member", "speaker", "label", "user", "person", "name"]
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    # If no label column exists, infer speaker from file_name
    if target_col is None:
        if "file_name" not in df.columns:
            raise ValueError(
                "No label column found and 'file_name' column is missing. "
                "Cannot infer speaker labels."
            )

        print("No explicit target column found. Inferring speaker from 'file_name'...")
        df["speaker"] = df["file_name"].apply(infer_speaker_from_filename)
        target_col = "speaker"

    print(f"Using '{target_col}' as target label.\n")

    # Drop non-feature columns (keep only numeric features)
    cols_to_drop = [
        target_col,
        "file_name",
        "file_path",
        "phrase",
        "is_authorized",
    ]

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    y = df[target_col]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}\n")

    # ---------------------------------------------
    # 3. Train / test split
    # ---------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    # ---------------------------------------------
    # 4. Train model
    # ---------------------------------------------
    print("Training RandomForest voice authentication model...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ---------------------------------------------
    # 5. Evaluate
    # ---------------------------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n Voice model trained.")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-score (weighted): {f1:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------
    # 6. Save model
    # ---------------------------------------------
    model_path = models_dir / "voice_authentication_model.pkl"
    joblib.dump(model, model_path)

    print(f"\n Model saved to: {model_path}")
    print("You can now use this model in authentication_system.py.")


if __name__ == "__main__":
    main()
