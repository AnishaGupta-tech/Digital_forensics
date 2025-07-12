import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging

class SignatureVerifier:
    def __init__(self, model_path='ai_models/signature_model.h5'):
        """
        Initialize the signature verifier with a pre-trained model.
        
        Args:
            model_path (str): Path to the saved .h5 model file
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the pre-trained model with error handling"""
        try:
            if not os.path.exists(self.model_path):
                error_msg = f"Model file not found at {self.model_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.model = load_model(self.model_path)
            self.logger.info(f"Successfully loaded model from {self.model_path}")
            
            # Verify model input shape
            expected_shape = (150, 150, 1)
            if self.model.input_shape[1:] != expected_shape:
                error_msg = f"Model expects input shape {self.model.input_shape[1:]}, but code is configured for {expected_shape}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def verify(self, image_path):
        """
        Verify if a signature is genuine or forged.
        
        Args:
            image_path (str): Path to the signature image file
            
        Returns:
            dict: {
                'authenticity_score': float (0-1),
                'verdict': 'GENUINE' or 'FORGED',
                'confidence': float (0-100),
                'error': str (if any)
            }
        """
        try:
            # Validate input file
            if not os.path.exists(image_path):
                error_msg = f"Image file not found: {image_path}"
                self.logger.error(error_msg)
                return {
                    'error': error_msg,
                    'verdict': 'ANALYSIS_FAILED'
                }

            # Load and preprocess image
            img = load_img(
                image_path,
                target_size=(150, 150),
                color_mode='grayscale'
            )
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = float(self.model.predict(img_array, verbose=0)[0][0])
            confidence = prediction if prediction > 0.5 else 1 - prediction
            verdict = 'GENUINE' if prediction >= 0.85 else 'FORGED'

            self.logger.info(f"Signature analysis complete - Score: {prediction:.2f}, Verdict: {verdict}")

            return {
                'authenticity_score': prediction,
                'verdict': verdict,
                'confidence': confidence * 100,
                'model_version': '1.0'
            }

        except Exception as e:
            error_msg = f"Signature verification failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'error': error_msg,
                'verdict': 'ANALYSIS_FAILED'
            }

    @classmethod
    def create_model(cls):
        """Model creation method (for training only)"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 1)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model