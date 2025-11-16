#!/usr/bin/env python3
"""
Multimodal Authentication System - Command Line Interface
This script demonstrates a complete authentication transaction:
1. Face Recognition: Grants access to the system
2. Voice Verification: Approves prediction request
3. Product Recommendation: Displays personalized suggestions
"""

import numpy as np
import pandas as pd
import joblib
import sys
import os

# LOAD ALL MODELS
def load_models():
    
    print("Loading models...", end=" ")
    
    try:
        models = {
            'face_model': joblib.load('Face_Recognition/models/face_recognition_model.pkl'),
            'face_scaler': joblib.load('Face_Recognition/models/face_scaler.pkl'),
            'face_encoder': joblib.load('Face_Recognition/models/face_label_encoder.pkl'),
            
            'voice_model': joblib.load('Audio processing/models/voice_authentication_model.pkl'),
            'voice_scaler': joblib.load('Audio processing/models/voice_scaler.pkl'),
            'voice_encoder': joblib.load('Audio processing/models/voice_label_encoder.pkl'),
            
            'product_model': joblib.load('models/product_recommendation_model.pkl'),
            'product_encoder': joblib.load('models/product_label_encoder.pkl'),
            'product_feature_encoders': joblib.load('models/product_feature_encoders.pkl')
        }
        
        print("Done")
        return models
        
    except FileNotFoundError as e:
        print(f"Error: Model file not found: {e}")
        sys.exit(1)

# STEP 1: FACE RECOGNITION
def recognize_face(face_features, models, confidence_threshold=0.70):
    """
    Recognize face from features
    
    Args:
        face_features: numpy array of facial features
        models: dict of loaded models
        confidence_threshold: minimum confidence for authentication
    
    Returns:
        dict with authentication result
    """
    print("STEP 1: FACE RECOGNITION")
    
    print("Analyzing facial features...")
    
    # Scale features
    features_scaled = models['face_scaler'].transform(face_features.reshape(1, -1))
    
    # Predict
    prediction = models['face_model'].predict(features_scaled)[0]
    probabilities = models['face_model'].predict_proba(features_scaled)[0]
    person_name = models['face_encoder'].inverse_transform([prediction])[0]
    confidence = probabilities.max()
    
    print(f"Best match: {person_name}")
    print(f"Confidence: {confidence:.2%}")
    
    # Authentication decision
    authenticated = confidence >= confidence_threshold
    if authenticated:
        print(f"FACE RECOGNIZED")
        print(f"Welcome, {person_name}!")
        print(f"Access granted to system")
    else:
        print(f"FACE NOT RECOGNIZED")
        print(f"Confidence too low: {confidence:.2%} < {confidence_threshold:.0%}")
        print(f"Access denied")
    return {
        'authenticated': authenticated,
        'name': person_name,
        'confidence': confidence
    }

# STEP 2: VOICE VERIFICATION
def verify_voice(voice_features, models, confidence_threshold=0.70):
    """
    Verify voice authorization
    
    Args:
        voice_features: numpy array of voice features
        models: dict of loaded models
        confidence_threshold: minimum confidence for verification
    
    Returns:
        dict with verification result
    """
    print("STEP 2: VOICE VERIFICATION")
    print("Analyzing voice sample...")
    
    # Scale features
    features_scaled = models['voice_scaler'].transform(voice_features.reshape(1, -1))
    
    # Predict (binary: authorized=1, unauthorized=0)
    prediction = models['voice_model'].predict(features_scaled)[0]
    probabilities = models['voice_model'].predict_proba(features_scaled)[0]
    speaker_name = models['voice_encoder'].inverse_transform([prediction])[0]
    confidence = probabilities.max()
    
    authorized = prediction == 1
    
    print(f"Voice signature analyzed")
    print(f"Confidence: {confidence:.2%}")
    
    # Verification decision
    verified = authorized and confidence >= confidence_threshold
    
    if verified:
        print(f"VOICE VERIFIED")
        print(f"Authorized user confirmed")
        print(f"Proceeding to product recommendation")
    else:
        print(f"VOICE VERIFICATION FAILED")
        if not authorized:
            print(f"Status: Unauthorized voice")
        else:
            print(f"Confidence too low: {confidence:.2%}")
        print(f"Transaction denied")
    
    return {
        'verified': verified,
        'confidence': confidence
    }

# STEP 3: PRODUCT RECOMMENDATION
def recommend_product(customer_features, models):
    """
    Recommend product for customer
    
    Args:
        customer_features: numpy array of customer features
        models: dict of loaded models
    
    Returns:
        dict with recommendation
    """
    print("STEP 3: PRODUCT RECOMMENDATION")
    print("Analyzing customer profile...")
    
    # Predict
    prediction = models['product_model'].predict(customer_features.reshape(1, -1))[0]
    probabilities = models['product_model'].predict_proba(customer_features.reshape(1, -1))[0]
    product = models['product_encoder'].inverse_transform([prediction])[0]
    confidence = probabilities.max()
    
    print(f"Processing purchase history...")
    print(f"Analyzing preferences...")
    
    print(f"RECOMMENDATION GENERATED")
    print(f"Recommended Product: {product}")
    print(f"Confidence: {confidence:.2%}")
    return {
        'product': product,
        'confidence': confidence
    }

# MAIN AUTHENTICATION FLOW
def run_authentication(face_features, voice_features, customer_features, models):
    """
    Complete authentication and recommendation flow
    
    Args:
        face_features: facial feature vector
        voice_features: voice feature vector
        customer_features: customer profile features
        models: loaded models
    
    Returns:
        dict with complete transaction result
    """
    print(" " * 15 + "MULTIMODAL AUTHENTICATION SYSTEM")
    # Step 1: Face Recognition
    face_result = recognize_face(face_features, models)
    
    if not face_result['authenticated']:
        print("TRANSACTION TERMINATED: Face recognition failed")
        return {'success': False, 'stage': 'face_recognition'}
    
    # Step 2: Voice Verification
    voice_result = verify_voice(voice_features, models)
    
    if not voice_result['verified']:
        print("TRANSACTION TERMINATED: Voice verification failed")
        return {'success': False, 'stage': 'voice_verification'}
    
    # Step 3: Product Recommendation
    product_result = recommend_product(customer_features, models)
    
    # Success
    print("TRANSACTION COMPLETED SUCCESSFULLY")
    print(f"User: {face_result['name']}")
    print(f"Voice Verified: Yes")
    print(f"Recommended: {product_result['product']}")
    
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
    """Run demonstration scenarios using real test data"""
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
        print(f"Customer features loaded: {product_features_df.shape}")
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Cannot run demonstration without test data")
        return
    
    # Scenario 1: Authorized Transaction (use first samples)
    print("SCENARIO 1: Authorized User - Complete Transaction")
    face_auth = face_features_df.iloc[0].values
    voice_auth = voice_features_df.iloc[0].values
    customer_auth = product_features_df.iloc[0].values
    
    result1 = run_authentication(face_auth, voice_auth, customer_auth, models)
    input("Press Enter to continue to next scenario...")
    
    # Scenario 2: Unauthorized Face (random features)
    print("SCENARIO 2: Unauthorized User - Face Not Recognized")
    np.random.seed(99)
    face_unauth = np.random.randn(len(face_auth)) * 2.0
    voice_any = voice_features_df.iloc[1].values if len(voice_features_df) > 1 else voice_auth
    customer_any = product_features_df.iloc[1].values if len(product_features_df) > 1 else customer_auth
    
    result2 = run_authentication(face_unauth, voice_any, customer_any, models)
    input("Press Enter to continue to next scenario...")
    
    # Scenario 3: Authorized Face but Unauthorized Voice (random voice)
    print("SCENARIO 3: Authorized Face but Unauthorized Voice ")
    face_auth2 = face_features_df.iloc[2].values if len(face_features_df) > 2 else face_auth
    np.random.seed(55)
    voice_unauth = np.random.randn(len(voice_auth)) * 2.0 - 1.0
    customer_auth2 = product_features_df.iloc[2].values if len(product_features_df) > 2 else customer_auth
    
    result3 = run_authentication(face_auth2, voice_unauth, customer_auth2, models)
    
    # Summary
    print(" " * 25 + "DEMO SUMMARY")
    print(f"Scenario 1 (Authorized): {'Success' if result1.get('success') else 'Failed'}")
    print(f"Scenario 2 (Unauthorized Face): {'Correctly Denied' if not result2.get('success') else 'Failed'}")
    print(f"Scenario 3 (Unauthorized Voice): {'Correctly Denied' if not result3.get('success') else 'Failed'}")

# MAIN ENTRY POINT
def main():
    """Main function"""
    
    print(" " * 10 + "MULTIMODAL AUTHENTICATION SYSTEM")
    
    # Load models
    models = load_models()
    
    # Run demonstration
    run_demo(models)
    
    print("System demonstration completed successfully")
    print("Thank you for using the Multimodal Authentication System!\n")

if __name__ == "__main__":
    main()