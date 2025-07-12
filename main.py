from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import joblib
import xgboost as xgb
import cv2
import numpy as np
from datetime import datetime
import uuid
import os
from typing import Dict, Any

app = FastAPI()

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Load models at startup
@app.on_event("startup")
async def load_models():
    app.state.models = {
        'signature': tf.keras.models.load_model('ai_models/signature/model.h5'),
        'ink': joblib.load('ai_models/ink/kmeans.pkl'),
        'metadata': joblib.load('ai_models/metadata/rf_model.pkl'),
        'ensemble': xgb.XGBClassifier()
    }
    app.state.models['ensemble'].load_model('ai_models/ensemble/xgb_model.json')

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        # 1. Save uploaded file temporarily with unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{str(uuid.uuid4())}{file_ext}"
        file_path = f"uploads/{unique_filename}"
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 2. Run all analyses
        signature_result = analyze_signature(file_path)
        ink_result = analyze_ink(file_path)
        metadata_result = analyze_metadata(file_path)
        
        # 3. Generate final verdict using ensemble model
        final_verdict = generate_verdict(
            signature_result['authenticity_score'],
            ink_result['consistency_score'],
            metadata_result['tamper_probability']
        )
        
        # 4. Clean up temporary file
        os.remove(file_path)
        
        return JSONResponse({
            "report_id": f"FSC-{str(uuid.uuid4())[:8]}",
            "analysis_date": datetime.now().isoformat(),
            "signature_analysis": signature_result,
            "ink_analysis": ink_result,
            "metadata_analysis": metadata_result,
            "final_verdict": final_verdict
        })
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def analyze_signature(file_path: str) -> Dict[str, Any]:
    """Analyze signature using our trained CNN model"""
    try:
        # Preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            file_path,
            target_size=(150, 150),
            color_mode='grayscale'
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        
        # Make prediction
        model = app.state.models['signature']
        prediction = model.predict(img_array)
        authenticity_score = float(prediction[0][0])
        
        # Determine verdict
        threshold = 0.85  # You can adjust this threshold
        verdict = "AUTHENTIC" if authenticity_score >= threshold else "FORGED"
        
        return {
            "authenticity_score": authenticity_score,
            "verdict": verdict,
            "model_confidence": float(np.max(prediction))
        }
        
    except Exception as e:
        return {
            "authenticity_score": 0.0,
            "verdict": "ANALYSIS_FAILED",
            "error": str(e)
        }

def analyze_ink(file_path: str) -> Dict[str, Any]:
    """Analyze ink consistency"""
    try:
        # Your existing ink analysis logic
        # This is just a placeholder implementation
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Could not read image file")
            
        # Example: Convert to LAB color space and analyze
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate ink consistency score (placeholder logic)
        consistency_score = float(np.mean(a) / 255)
        
        return {
            "consistency_score": consistency_score,
            "verdict": "SINGLE_INK" if consistency_score > 0.8 else "MULTIPLE_INKS"
        }
        
    except Exception as e:
        return {
            "consistency_score": 0.0,
            "verdict": "ANALYSIS_FAILED",
            "error": str(e)
        }

def analyze_metadata(file_path: str) -> Dict[str, Any]:
    """Analyze document metadata"""
    try:
        # Your existing metadata analysis logic
        # This is just a placeholder implementation
        file_stats = os.stat(file_path)
        tamper_probability = 0.1  # Placeholder value
        
        return {
            "tamper_probability": tamper_probability,
            "verdict": "AUTHENTIC" if tamper_probability < 0.5 else "TAMPERED",
            "file_size": file_stats.st_size,
            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat()
        }
        
    except Exception as e:
        return {
            "tamper_probability": 1.0,
            "verdict": "ANALYSIS_FAILED",
            "error": str(e)
        }

def generate_verdict(sig_score: float, ink_score: float, meta_score: float) -> Dict[str, Any]:
    """Generate final verdict using ensemble model"""
    try:
        # Prepare input features for ensemble model
        features = np.array([[sig_score, ink_score, meta_score]])
        
        # Get prediction from ensemble model
        ensemble_model = app.state.models['ensemble']
        prediction = ensemble_model.predict(features)
        confidence = float(np.max(ensemble_model.predict_proba(features)))
        
        verdict_map = {
            0: "DOCUMENT_AUTHENTIC",
            1: "DOCUMENT_FORGED",
            2: "DOCUMENT_SUSPICIOUS"
        }
        
        return {
            "verdict": verdict_map.get(int(prediction[0]), "UNKNOWN"),
            "confidence": confidence * 100,
            "model_used": "XGBoost Ensemble"
        }
        
    except Exception as e:
        return {
            "verdict": "ANALYSIS_FAILED",
            "error": str(e)
        }