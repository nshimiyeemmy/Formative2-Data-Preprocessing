import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


def main():
    # 1. Locate project paths
    script_dir = Path(__file__).resolve().parent      # .../scripts
    project_root = script_dir.parent                  # project root

    features_path = project_root / "Face_Recognition" / "image_features.csv"
    models_dir = project_root / "Face_Recognition" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Loading image features from: {features_path}")

    if not features_path.exists():
        print(" image_features.csv not found. Make sure it exists in Face_Recognition/")
        return

    # 2. Load dataset
    df = pd.read_csv(features_path)

    # Try to detect the target column
    possible_targets = ["member_name", "member", "label", "person", "user"]
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            f"Could not find a target column. "
            f"Looked for: {possible_targets}. "
            f"Columns found: {list(df.columns)}"
        )

    print(f"Using '{target_col}' as target label.\n")


    cols_to_drop = [
        target_col,
        "filename",
        "file_name",
        "image_path",
        "expression",
        "folder",
    ]

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    y = df[target_col]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}\n")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 4. Train model
    print("Training RandomForest face recognition model...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n Face model trained.")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-score (weighted): {f1:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Save model
    model_path = models_dir / "face_recognition_model.pkl"
    joblib.dump(model, model_path)

    print(f"\n Model saved to: {model_path}")
    print("You can now use this model in authentication_system.py.")


if __name__ == "__main__":
    main()
