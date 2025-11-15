# Formative 2: Multimodal Data Preprocessing & Recommendation System

## 📌 Assignment Overview

This project implements a multimodal authentication and product recommendation workflow. The system authenticates a user through **facial recognition** and **voice validation** before granting access to the product recommendation model.

**Flow summary:**
1. User attempts access  
2. Face recognition → if unrecognized → access denied  
3. Voice verification → if mismatch → access denied  
4. Verified user → allowed to run product prediction  
5. Recommended product is displayed  

# Data Merge (Tabular Data)

Two datasets are used:  
- `customer_social_profiles`  
- `customer_transactions`

### Tasks
- Merge both datasets on the customer ID  
- Preprocess & clean data  
- Engineer features (e.g., interactions, spending patterns)  
- Produce a final ML-ready dataset  

✔ Output: **merged_customer_data.csv**



#  Image Data Collection & Processing

Each member submits **3 face images**:
- Neutral  
- Smiling  
- Surprised  

### Required Steps
- Load & display images  
- Apply augmentations:
  - Rotation  
  - Flip  
  - Grayscale, etc.  
- Extract image features:
  - Embeddings  
  - Histograms  
  - Texture features  

✔ Output: **image_features.csv**

---

#  Audio Data Collection & Processing

Each member records **2 audio samples** saying:
- “Yes, approve”  
- “Confirm transaction”

### Required Steps
- Load & visualize:
  - Waveforms  
  - Spectrograms  
- Apply augmentations:
  - Pitch shift  
  - Time stretch  
  - Background noise  
- Extract features:
  - MFCCs  
  - Spectral roll-off  
  - Energy, zero-crossing rate  

✔ Output: **audio_features.csv**


# Model Creation


###  Facial Recognition Model
- Uses image embeddings  
- Predict whether the face matches a registered user  

###  Voiceprint Verification Model
- Uses MFCC-based audio features  
- Predict whether the voice matches the user  

###  Product Recommendation Model
- Based on merged tabular dataset  
- Predicts likely product purchase  

### Suggested Algorithms
- Random Forest  
- Logistic Regression  
- XGBoost  
- SVM  

### Evaluation Metrics
- Accuracy  
- F1-Score  
- Loss  


### Simulations Required
✔ Unauthorized face attempt  
✔ Unauthorized audio attempt  

### Full authorized transaction:
1. Input face → system verifies identity  
2. If recognized → user allowed to request prediction  
3. Input voice → system verifies approval phrase  
4. Product recommendation displayed  



###  Datasets
- Raw datasets  
- Merged dataset  
- Image and audio samples 

### Engineered Features
- `merged_customer_data.csv`  
- `image_features.csv`  
- `audio_features.csv`

### 🔹 Code
- Image preprocessing script  
- Audio preprocessing script  
- Multimodal feature engineering script  
- Facial recognition model  
- Voice verification model  
- Product recommendation model  
- Command-line app script  
- Jupyter Notebook containing:
  - EDA  
  - Preprocessing steps  
  - Model training  
  - Evaluation results  

# 👥 Team Contributions
task1: Emmy Nshimiye
Task2: Tracy Umukunzi
Task3: Carine Ahishakiye
Task4: Eva Nice 



