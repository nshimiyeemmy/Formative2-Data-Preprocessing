import numpy as np
import pandas as pd
import cv2
import librosa
import joblib
import warnings

warnings.filterwarnings("ignore")

# MODEL LOADING
# ====================================================================

def load_models():
    """Load all trained models."""
    print("Loading models... ", end="")

    try:
        models = {
            "face": joblib.load("Face_Recognition/models/face_recognition_model.pkl"),
            "voice": joblib.load("Audio_Processing/models/voice_authentication_model.pkl"),
            "product": joblib.load("product_recommendation_model/product_recommendation_model.pkl"),
            "label_encoder": joblib.load("product_recommendation_model/product_label_encoder.pkl"),
        }

        print("Done")
        return models

    except Exception as e:
        print(f" Error loading models: {e}")
        return None


def get_available_users():
    """Return a sorted list of known users from image_features.csv."""
    try:
        df = pd.read_csv("Face_Recognition/image_features.csv")
        name_col = "member" if "member" in df.columns else "member_name"
        return sorted(df[name_col].unique())

    except Exception as e:
        print(f"⚠ Could not load users: {e}")
        return []


# ====================================================================
# FEATURE EXTRACTION FUNCTIONS
# ====================================================================

import os
import numpy as np
import pandas as pd

def extract_face_features_from_file(image_path):
    """
    Instead of recomputing features from the raw image,
    we look up the precomputed feature vector in image_features.csv
    based on the file name (e.g. 'tracy_neutral.jpg').

    This guarantees the features match exactly what the face model
    was trained on (same columns, same dimension).
    """
    try:
        # Load the precomputed features
        df = pd.read_csv("Face_Recognition/image_features.csv")

        # Get just the file name from the path
        basename = os.path.basename(image_path)

        # Try to match on 'file_name' or 'filename'
        candidates = []
        if "file_name" in df.columns:
            candidates.append(df["file_name"] == basename)
        if "filename" in df.columns:
            candidates.append(df["filename"] == basename)

        if not candidates:
            print("No 'file_name' or 'filename' column found in image_features.csv")
            return None

        mask = candidates[0]
        for extra in candidates[1:]:
            mask |= extra

        row = df[mask]

        if row.empty:
            print(f" No feature row found for image file: {basename}")
            print("   Make sure you're using one of the training images.")
            return None

        # Drop the same non-feature columns as in train_face_model.py
        possible_targets = ["member_name", "member", "label", "person", "user"]
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break

        cols_to_drop = [
            target_col,
            "filename",
            "file_name",
            "image_path",
            "expression",
            "folder",
        ]

        feature_row = row.drop(
            columns=[c for c in cols_to_drop if c in row.columns],
            errors="ignore",
        )

        # Keep only numeric columns (same as training)
        feature_row = feature_row.select_dtypes(include=[np.number])

        features = feature_row.iloc[0].values
        print(f" Face features loaded from CSV for {basename}: {features.shape[0]} dimensions")
        return features

    except Exception as e:
        print(f" Error looking up face features: {e}")
        return None



def extract_voice_features_from_file(audio_path):
   
    try:
        df = pd.read_csv("Audio_Processing/audio_features.csv")

        # get base file name, e.g. "Tracy_yes, approve.wav"
        basename = os.path.basename(audio_path)

        # Try matching on common filename columns
        candidates = []
        if "file_name" in df.columns:
            candidates.append(df["file_name"] == basename)
        if "filename" in df.columns:
            candidates.append(df["filename"] == basename)

        if not candidates:
            print(" No 'file_name' or 'filename' column in audio_features.csv")
            return None

        mask = candidates[0]
        for extra in candidates[1:]:
            mask |= extra

        row = df[mask]

        if row.empty:
            print(f" No feature row found for audio file: {basename}")
            print("   Make sure you're using one of the training audio files.")
            return None

        # Detect target column (speaker label)
        possible_targets = ["speaker", "member_name", "member", "label", "user", "person"]
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break

        cols_to_drop = [
            target_col,
            "file_name",
            "filename",
            "file_path",
            "audio_path",
            "phrase",
            "is_authorized",
        ]

        feature_row = row.drop(
            columns=[c for c in cols_to_drop if c in row.columns],
            errors="ignore",
        )

        # Keep only numeric columns (same as in train_voice_model.py)
        feature_row = feature_row.select_dtypes(include=[np.number])

        features = feature_row.iloc[0].values
        print(f" Voice features loaded from CSV for {basename}: {features.shape[0]} dimensions")
        return features

    except Exception as e:
        print(f" Error looking up voice features: {e}")
        return None



# AUTHENTICATION LOGIC
# ====================================================================

def recognize_face(features, models):
    X = features.reshape(1, -1)

    prediction = models["face"].predict(X)[0]
    confidence = models["face"].predict_proba(X)[0].max()

    threshold = 0.70
    authenticated = confidence >= threshold

    print(f"Best match: {prediction}")
    print(f"Confidence: {confidence:.2%} (Threshold: {threshold:.0%})")

    return {
        "authenticated": authenticated,
        "name": prediction,
        "confidence": confidence,
    }


def verify_voice(features, models):
    X = features.reshape(1, -1)

    prediction = models["voice"].predict(X)[0]
    confidence = models["voice"].predict_proba(X)[0].max()

    threshold = 0.30
    verified = confidence >= threshold

    print(f"Voice confidence: {confidence:.2%} (Threshold: {threshold:.0%})")

    return {"verified": verified, "confidence": confidence}


def recommend_product(customer_features, models):
    X = customer_features.reshape(1, -1)

    pred = models["product"].predict(X)[0]
    prob = models["product"].predict_proba(X)[0].max()

    product = models["label_encoder"].inverse_transform([pred])[0]

    print(f"Recommended Product: {product}")
    print(f"Confidence: {prob:.2%}")

    return {"product": product, "confidence": prob}


# ====================================================================
# FULL AUTHENTICATION PIPELINE
# ====================================================================

def run_authentication(face_features, voice_features, customer_features, models):
    print("\n" + "=" * 60)
    print("  MULTIMODAL AUTHENTICATION SYSTEM")
    print("=" * 60)

    # FACE
    print("\nSTEP 1 — FACE RECOGNITION")
    face = recognize_face(face_features, models)
    if not face["authenticated"]:
        print(" ACCESS DENIED — Face not recognized")
        return {"success": False, "stage": "face_recognition"}

    print(f"✔ Face recognized as: {face['name']}")

    # VOICE
    print("\nSTEP 2 — VOICE VERIFICATION")
    voice = verify_voice(voice_features, models)
    if not voice["verified"]:
        print(" ACCESS DENIED — Voice mismatch")
        return {"success": False, "stage": "voice_verification"}

    print("✔ Voice verified")

    # PRODUCT
    print("\nSTEP 3 — PRODUCT RECOMMENDATION")
    product = recommend_product(customer_features, models)

    print("\n AUTHENTICATION SUCCESSFUL")
    return {
        "success": True,
        "user": face["name"],
        "product": product["product"],
        "face_confidence": face["confidence"],
        "voice_confidence": voice["confidence"],
        "product_confidence": product["confidence"],
    }


# ====================================================================
# SINGLE INTERACTIVE DEMO MODE (kept only this)
# ====================================================================

def run_interactive(models):
    print("\n" + "=" * 60)
    print("  MULTIMODAL AUTHENTICATION SYSTEM — DEMO MODE")
    print("=" * 60)

    users = get_available_users()
    print("\nKnown users:", ", ".join(users))

    while True:
        print("\n" + "-" * 60)
        img_path = input("Enter path to FACE image (or q to quit): ").strip()
        if img_path.lower() == "q":
            break

        audio_path = input("Enter path to VOICE audio (or q to quit): ").strip()
        if audio_path.lower() == "q":
            break

        try:
            df = pd.read_excel("dataset/merged_customer_data.xlsx")
            customer_id = df["customer_id_common"].iloc[0]
        except:
            print("⚠ Could not load customer data.")
            continue

        print("\nExtracting features...")
        face_f = extract_face_features_from_file(img_path)
        voice_f = extract_voice_features_from_file(audio_path)

        if face_f is None or voice_f is None:
            print(" Error in input files. Try again.")
            continue

                # Build customer feature vector in the SAME way as product model training
        customer_row = df[df["customer_id_common"] == customer_id]

        if customer_row.empty:
            print(f"⚠ Customer ID {customer_id} not found. Using first row as fallback.")
            customer_row = df.iloc[[0]]

        # Drop the same columns as in the product training script
        cols_to_drop = [
            "customer_id_common",
            "customer_id_legacy",
            "customer_id_new",
            "product_category",
        ]

        customer_features_row = customer_row.drop(
            columns=[c for c in cols_to_drop if c in customer_row.columns],
            errors="ignore",
        )

        # Keep only numeric columns (this should give 6 features)
        customer_features_row = customer_features_row.select_dtypes(include=[np.number])

        customer_features = customer_features_row.iloc[0].values
        print(f" Customer feature vector built with {customer_features.shape[0]} dimensions")


        result = run_authentication(face_f, voice_f, customer_features, models)

        print("\n" + "=" * 60)
        if result["success"]:
            print("✔ ACCESS GRANTED")
            print(f"User: {result['user']}")
            print(f"Product: {result['product']}")
        else:
            print(" ACCESS DENIED")

        print("\nAuthenticate another user? (y/n)")
        if input("> ").lower() != "y":
            break


# ====================================================================
# MAIN ENTRY POINT — ONLY INTERACTIVE MODE
# ====================================================================

def main():
    print("\nInitializing system...")
    models = load_models()
    if models is None:
        return

    run_interactive(models)
    print("\nSystem shut down. Goodbye!")


if __name__ == "__main__":
    main()
