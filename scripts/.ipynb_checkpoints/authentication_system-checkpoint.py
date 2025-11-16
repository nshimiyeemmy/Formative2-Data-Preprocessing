import numpy as np
import pandas as pd
import cv2
import librosa
from PIL import Image
import joblib
import warnings
warnings.filterwarnings('ignore')

# MODEL LOADING FUNCTION (MISSING IN YOUR CODE)
def load_models():
    """Load all trained models"""
    print("Loading models... ", end="")
    
    try:
        models = {
            'face': joblib.load('Face_Recognition/models/face_recognition_model.pkl'),
            'voice': joblib.load('models/voice_verification_model.pkl'),
            'product': joblib.load('models/product_recommender_model.pkl'),
            'label_encoder': joblib.load('models/label_encoder.pkl')
        }
        print("Done")
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# FILE PROCESSING FUNCTIONS
def extract_face_features_from_file(image_path):
    """
    Extract features from a new image file
    
    Args:
        image_path: path to the image file
        
    Returns:
        feature vector matching training data format
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size (adjust based on your training)
        img_resized = cv2.resize(img_rgb, (128, 128))
        
        # Extract features (histogram-based - adjust to match your training)
        # This is a simple example - replace with your actual feature extraction
        hist_r = cv2.calcHist([img_resized], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_resized], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_resized], [2], None, [32], [0, 256]).flatten()
        
        # Combine features
        features = np.concatenate([hist_r, hist_g, hist_b])
        
        # Normalize
        features = features / (features.sum() + 1e-7)
        
        print(f"Face features extracted: {len(features)} dimensions")
        return features
        
    except Exception as e:
        print(f"Error extracting face features: {e}")
        return None


def extract_voice_features_from_file(audio_path):
    """
    Extract features from a new audio file
    
    Args:
        audio_path: path to the audio file
        
    Returns:
        feature vector matching training data format
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract MFCCs (13 coefficients, typical)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Extract spectral features
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Extract energy (RMS)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Combine features
        features = np.concatenate([
            mfccs_mean,
            [spectral_rolloff, spectral_centroid, rms]
        ])
        
        print(f"Voice features extracted: {len(features)} dimensions")
        return features
        
    except Exception as e:
        print(f"Error extracting voice features: {e}")
        return None

# AUTHENTICATION FUNCTIONS (MISSING IN YOUR CODE)

def recognize_face(face_features, models):
    """
    Recognize face from features
    
    Args:
        face_features: facial feature vector
        models: loaded models dict
        
    Returns:
        dict with authentication result
    """
    # Reshape for prediction
    X = face_features.reshape(1, -1)
    
    # Predict
    prediction = models['face'].predict(X)[0]
    probabilities = models['face'].predict_proba(X)[0]
    confidence = probabilities.max()
    
    # Threshold for authentication
    threshold = 0.70
    authenticated = confidence >= threshold
    
    print(f"Best match: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    
    if authenticated:
        print("FACE RECOGNIZED")
    else:
        print("FACE NOT RECOGNIZED")
    
    return {
        'authenticated': authenticated,
        'name': prediction,
        'confidence': confidence
    }


def verify_voice(voice_features, models):
    """
    Verify voice from features
    
    Args:
        voice_features: voice feature vector
        models: loaded models dict
        
    Returns:
        dict with verification result
    """
    # Reshape for prediction
    X = voice_features.reshape(1, -1)
    
    # Predict
    prediction = models['voice'].predict(X)[0]
    probabilities = models['voice'].predict_proba(X)[0]
    confidence = probabilities.max()
    
    # For voice, we can use a lower threshold
    threshold = 0.30
    verified = confidence >= threshold
    
    print(f"Voice signature analyzed")
    print(f"Confidence: {confidence:.2%}")
    
    if verified:
        print("VOICE VERIFICATION PASSED")
    else:
        print("VOICE VERIFICATION FAILED")
        print("Status: Unauthorized voice")
    
    return {
        'verified': verified,
        'confidence': confidence
    }


def recommend_product(customer_features, models):
    """
    Recommend product based on customer profile
    
    Args:
        customer_features: customer feature vector
        models: loaded models dict
        
    Returns:
        dict with product recommendation
    """
    print("Processing purchase history...")
    print("Analyzing preferences...")
    
    # Reshape for prediction
    X = customer_features.reshape(1, -1)
    
    # Predict
    prediction = models['product'].predict(X)[0]
    probabilities = models['product'].predict_proba(X)[0]
    
    # Decode product name
    product = models['label_encoder'].inverse_transform([prediction])[0]
    confidence = probabilities.max()
    
    print("RECOMMENDATION GENERATED")
    print(f"Recommended Product: {product}")
    print(f"Confidence: {confidence:.2%}")
    
    return {
        'product': product,
        'confidence': confidence
    }

# CUSTOM FILE TESTING

def test_custom_files(image_path, audio_path, customer_id, models):
    print(" " * 20 + "CUSTOM FILE AUTHENTICATION TEST")
    
    # Extract face features
    print("Processing face image...")
    face_features = extract_face_features_from_file(image_path)
    if face_features is None:
        print("AUTHENTICATION FAILED: Could not process face image")
        print("ACCESS DENIED!")
        return {'success': False, 'stage': 'image_processing'}
    
    # Extract voice features
    print("Processing voice sample...")
    voice_features = extract_voice_features_from_file(audio_path)
    if voice_features is None:
        print("AUTHENTICATION FAILED: Could not process voice sample")
        print("ACCESS DENIED!")
        return {'success': False, 'stage': 'audio_processing'}
    
    # Get customer features
    print("Loading customer profile...")
    try:
        customer_df = pd.read_excel('dataset/merged_customer_data.xlsx')
        customer_data = customer_df[customer_df['customer_id_common'] == customer_id]
        
        if customer_data.empty:
            print(f"Customer ID {customer_id} not found, using default profile")
            customer_data = customer_df.iloc[0:1]
        
        # Prepare features
        customer_cols_to_drop = ['customer_id_common', 'customer_id_legacy', 
                                'customer_id_new', 'product_category']
        customer_features = customer_data.drop(
            columns=[col for col in customer_cols_to_drop if col in customer_data.columns], 
            errors='ignore'
        )
        customer_features = customer_features.select_dtypes(include=[np.number]).iloc[0].values
        
    except Exception as e:
        print(f"Error loading customer data: {e}")
        print("TRANSACTION TERMINATED!")
        return {'success': False, 'stage': 'customer_data'}
    
    # Run authentication flow
    result = run_authentication(face_features, voice_features, customer_features, models)
    
    return result

# MAIN AUTHENTICATION FLOW

def run_authentication(face_features, voice_features, customer_features, models):
    """
    Complete authentication and recommendation flow
    """
    print(" " * 15 + "MULTIMODAL AUTHENTICATION SYSTEM")
    
    # Step 1: Face Recognition
    print("STEP 1: FACE RECOGNITION")
    print("Analyzing facial features...")
    face_result = recognize_face(face_features, models)
    
    if not face_result['authenticated']:
        print("FACE NOT RECOGNIZED")
        print(f"Confidence too low: {face_result['confidence']:.2%} < 70%")
        print("ACCESS DENIED!")
        print("TRANSACTION TERMINATED: Face recognition failed")
        return {'success': False, 'stage': 'face_recognition'}
    
    print(f"FACE RECOGNIZED")
    print(f"Welcome, {face_result['name']}!")
    print("Access granted to system")
    
    # Step 2: Voice Verification
    print("STEP 2: VOICE VERIFICATION")
    print("Analyzing voice sample...")
    voice_result = verify_voice(voice_features, models)
    
    if not voice_result['verified']:
        print("VOICE VERIFICATION FAILED")
        print(f"Status: Unauthorized voice")
        print("Transaction denied")
        print("TRANSACTION TERMINATED: Voice verification failed")
        return {'success': False, 'stage': 'voice_verification'}
    
    print(f"Voice signature analyzed")
    print(f"Confidence: {voice_result['confidence']:.2%}")
    print("VOICE AUTHENTICATED")
    
    # Step 3: Product Recommendation
    print("STEP 3: PRODUCT RECOMMENDATION")
    product_result = recommend_product(customer_features, models)
    
    # Success
    print("TRANSACTION COMPLETED SUCCESSFULLY")
    print(f"User: {face_result['name']}")
    print(f"Voice Verified: Yes")
    print(f"Recommended Product: {product_result['product']}")
    print(f"Recommendation Confidence: {product_result['confidence']:.2%}")
    
    return {
        'success': True,
        'user': face_result['name'],
        'product': product_result['product'],
        'face_confidence': face_result['confidence'],
        'voice_confidence': voice_result['confidence'],
        'product_confidence': product_result['confidence']
    }

# DEMO SCENARIOS

def run_demo(models):
    """Run demonstration scenarios including custom file upload option"""
    print(" " * 20 + "SYSTEM DEMONSTRATION")
    
    # Load real test data
    print("Loading test data...")
    
    try:
        # Load image features
        image_df = pd.read_csv('Face_Recognition/image_features.csv')
        image_cols_to_drop = ['filename', 'member_name', 'expression', 'image_path', 'file_name']
        face_features_df = image_df.drop(columns=[col for col in image_cols_to_drop if col in image_df.columns], errors='ignore')
        face_features_df = face_features_df.select_dtypes(include=[np.number])
        
        # Load audio features
        audio_df = pd.read_csv('Audio processing/audio_features.csv')
        audio_cols_to_drop = ['member_name', 'phrase', 'is_authorized', 'audio_path', 'file_name', 'filename']
        voice_features_df = audio_df.drop(columns=[col for col in audio_cols_to_drop if col in audio_df.columns], errors='ignore')
        voice_features_df = voice_features_df.select_dtypes(include=[np.number])
        
        # Load customer features
        customer_df = pd.read_excel('dataset/merged_customer_data.xlsx')
        customer_cols_to_drop = ['customer_id_common', 'customer_id_legacy', 'customer_id_new', 'product_category']
        product_features_df = customer_df.drop(columns=[col for col in customer_cols_to_drop if col in customer_df.columns], errors='ignore')
        product_features_df = product_features_df.select_dtypes(include=[np.number])
        
        print(f"Face features loaded: {face_features_df.shape}")
        print(f"Voice features loaded: {voice_features_df.shape}")
        print(f"Customer features loaded: {product_features_df.shape}\n")
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Cannot run demonstration without test data")
        return
    
    # Scenario 1: Authorized Transaction
    print("SCENARIO 1: Authorized User - Complete Transaction")
    face_auth = face_features_df.iloc[0].values
    voice_auth = voice_features_df.iloc[0].values
    customer_auth = product_features_df.iloc[0].values
    
    result1 = run_authentication(face_auth, voice_auth, customer_auth, models)
    input("Press Enter to continue to next scenario...")
    
    # Scenario 2: Unauthorized Face
    print("SCENARIO 2: Unauthorized Face - Access Denied")
    np.random.seed(99)
    face_unauth = np.random.randn(len(face_auth)) * 2.0
    voice_any = voice_features_df.iloc[1].values if len(voice_features_df) > 1 else voice_auth
    customer_any = product_features_df.iloc[1].values if len(product_features_df) > 1 else customer_auth
    
    result2 = run_authentication(face_unauth, voice_any, customer_any, models)
    input("Press Enter to continue to next scenario...")
    
    # Scenario 3: Unauthorized Voice
    print("SCENARIO 3: Unauthorized Voice - Transaction Denied")
    face_auth2 = face_features_df.iloc[2].values if len(face_features_df) > 2 else face_auth
    np.random.seed(55)
    voice_unauth = np.random.randn(len(voice_auth)) * 2.0 - 1.0
    customer_auth2 = product_features_df.iloc[2].values if len(product_features_df) > 2 else customer_auth
    
    result3 = run_authentication(face_auth2, voice_unauth, customer_auth2, models)
    
    # Summary
    print(" " * 25 + "DEMO SUMMARY")
    print(f"Scenario 1 (Authorized): {'Success' if result1.get('success') else 'Failed'}")
    print(f"Scenario 2 (Unauthorized Face): {'Access Denied' if not result2.get('success') else 'Failed'}")
    print(f"Scenario 3 (Unauthorized Voice): {'Access Denied' if not result3.get('success') else 'Failed'}")
    
    # OPTION FOR CUSTOM FILE TESTING
    print("Would you like to test with your own image and audio files? (y/n)")
    choice = input("Enter choice: ").strip().lower()
    
    if choice == 'y':
        print("Custom File Testing")
        image_path = input("Enter path to face image (e.g., 'test_face.jpg'): ").strip()
        audio_path = input("Enter path to voice audio (e.g., 'test_voice.wav'): ").strip()
        customer_id = input("Enter customer ID (or press Enter for default): ").strip()
        
        if not customer_id:
            customer_id = customer_df['customer_id_common'].iloc[0]
        
        # Test with custom files
        custom_result = test_custom_files(image_path, audio_path, customer_id, models)
        
        print("CUSTOM FILE TEST RESULT:")
        if custom_result.get('success'):
            print("Authentication Successful!")
            print(f"User: {custom_result.get('user', 'Unknown')}")
            print(f"Product: {custom_result.get('product', 'N/A')}")
        else:
            print("Authentication Failed")
            print(f"Failed at: {custom_result.get('stage', 'unknown')}")

# MAIN ENTRY POINT
def main():
    print(" " * 10 + "MULTIMODAL AUTHENTICATION SYSTEM")
    
    # Load models
    print("Loading models...")
    models = load_models()
    
    if models is None:
        print("Failed to load models. Exiting.")
        return
    
    # Run demonstration
    run_demo(models)
    
    print("System demonstration completed successfully")
    print("Thank you for using the Multimodal Authentication System!")


if __name__ == "__main__":
    main()
