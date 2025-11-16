import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import joblib


def main():
    # ---------------------------------------------
    # 1. Locate paths
    # ---------------------------------------------
    script_dir = Path(__file__).resolve().parent   # .../scripts
    project_root = script_dir.parent               # project root

    data_path = project_root / "dataset" / "merged_customer_data.xlsx"
    model_dir = project_root / "product_recommendation_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Loading merged data from: {data_path}")

    if not data_path.exists():
        print("❌ merged_customer_data.xlsx not found in dataset/")
        return

    # ---------------------------------------------
    # 2. Load dataset
    # ---------------------------------------------
    df = pd.read_excel(data_path)

    # Try to detect the target column for product
    possible_targets = ["product_category", "product", "product_name", "purchased_product"]
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            f"Could not find a target column. "
            f"Looked for: {possible_targets}. "
            f"Columns: {list(df.columns)}"
        )

    print(f"Using '{target_col}' as target label.\n")

    # Drop ID and target columns → keep same style as authentication_system.py
    cols_to_drop = [
        "customer_id_common",
        "customer_id_legacy",
        "customer_id_new",
        target_col,
    ]

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    y = df[target_col]

    print(f"Feature matrix shape: {X.shape}")   # this should be (213, 6)
    print(f"Target distribution:\n{y.value_counts()}\n")

    # ---------------------------------------------
    # 3. Encode target labels
    # ---------------------------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ---------------------------------------------
    # 4. Train / test split
    # ---------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    # ---------------------------------------------
    # 5. Train XGBoost model
    # ---------------------------------------------
    print("Training XGBoost product recommendation model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------
    # 6. Evaluate
    # ---------------------------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n Product model trained.")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-score (weighted): {f1:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------
    # 7. Save model + label encoder
    # ---------------------------------------------
    model_path = model_dir / "product_recommendation_model.pkl"
    label_path = model_dir / "product_label_encoder.pkl"

    joblib.dump(model, model_path)
    joblib.dump(le, label_path)

    print(f"\n Model saved to: {model_path}")
    print(f" Label encoder saved to: {label_path}")
    print("authentication_system.py can now use these for product prediction.")


if __name__ == "__main__":
    main()
