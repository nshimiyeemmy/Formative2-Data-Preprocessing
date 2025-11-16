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

def extract_face_features_from_file(image_path, models):
    """Extract face features and match the model's expected input size."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))

        # Training used 256 histogram bins per channel
        hist_r = cv2.calcHist([img_resized], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_resized], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_resized], [2], None, [256], [0, 256]).flatten()

        features = np.concatenate([hist_r, hist_g, hist_b])
        features = features / (np.sum(features) + 1e-7)

        # Pad or trim to match model dimension
        required_dim = models["face"].n_features_in_

        if len(features) < required_dim:
            features = np.pad(features, (0, required_dim - len(features)), constant_values=0)
        elif len(features) > required_dim:
            features = features[:required_dim]

        print(f"Face features extracted: {len(features)} dims (expected {required_dim})")
        return features

    except Exception as e:
        print(f" Face extraction failed: {e}")
        return None


def extract_voice_features_from_file(audio_path):
    """Extract MFCC + spectral features."""
    try:
        y, sr = librosa.load(audio_path, sr=22050)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))

        features = np.concatenate([mfccs_mean, [rolloff, centroid, rms]])

        print(f"Voice features extracted: {len(features)} dimensions")
        return features

    except Exception as e:
        print(f" Voice extraction failed: {e}")
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
        face_f = extract_face_features_from_file(img_path, models)
        voice_f = extract_voice_features_from_file(audio_path)

        if face_f is None or voice_f is None:
            print(" Error in input files. Try again.")
            continue

        customer_row = df[df["customer_id_common"] == customer_id]
        customer_features = (
            customer_row.select_dtypes(include=[np.number]).iloc[0].values
        )

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
