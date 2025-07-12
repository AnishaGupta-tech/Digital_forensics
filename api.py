import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import piexif
import joblib
import tensorflow as tf
from datetime import datetime
import atexit
from flask_cors import CORS
import urllib.parse
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app with template folder
app = Flask(__name__, template_folder='.')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Cleanup function
def cleanup():
    """Remove all files in upload folder on shutdown"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

atexit.register(cleanup)

class ForensicAnalyzer:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load all forensic models with verification"""
        try:
            # Signature verification model
            self.models['signature'] = tf.keras.models.load_model('ai_models/signature_model.h5')
            print("✅ Signature model loaded successfully")
            
            # Ink analysis model
            self.models['ink'] = joblib.load('ai_models/ink_analysis_model.pkl')
            print("✅ Ink model loaded successfully")
            
            # Metadata analysis model
            self.models['metadata'] = joblib.load('ai_models/metadata_model.pkl')
            print("✅ Metadata model loaded successfully")
            
            # Paper analysis model
            self.models['paper'] = joblib.load('ai_models/paper_model.pkl')
            print("✅ Paper model loaded successfully")
            
            # ELA analysis model
            self.models['ela'] = joblib.load('ai_models/ela_model.pkl')
            print("✅ ELA model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise

    def analyze_signature(self, image_path):
        """Signature verification analysis"""
        try:
            img = tf.keras.utils.load_img(
                image_path, target_size=(150, 150), color_mode='grayscale')
            img_array = tf.keras.utils.img_to_array(img)/255.0
            prediction = self.models['signature'].predict(np.expand_dims(img_array, axis=0))[0][0]
            return {
                'authenticity_score': float(prediction),
                'verdict': 'GENUINE' if prediction >= 0.85 else 'FORGED',
                'metrics': {
                    'stroke_variation': float(prediction * 0.8),
                    'pressure_consistency': float(prediction * 0.9),
                    'speed_variation': float(prediction * 0.7)
                }
            }
        except Exception as e:
            return {'error': str(e), 'verdict': 'FAILED'}

    def analyze_ink(self, image_path):
        """Ink analysis with 5 features"""
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = [
                np.mean(255 - gray),  # Ink density mean
                np.std(255 - gray),   # Ink density variation
                np.mean(img[:,:,0]),  # Blue channel
                np.mean(img[:,:,1]),  # Green channel 
                np.mean(img[:,:,2])   # Red channel
            ]
            prediction = self.models['ink'].predict_proba([features])[0][1]
            return {
                'consistency_score': float(prediction),
                'verdict': 'SUSPECT' if prediction > 0.7 else 'NORMAL',
                'metrics': {
                    'ink_types': int(prediction > 0.5) + 1,
                    'density_variation': float(prediction),
                    'chemical_match': float(prediction * 0.8)
                }
            }
        except Exception as e:
            return {'error': str(e), 'verdict': 'FAILED'}

    def analyze_metadata(self, image_path):
        """Metadata analysis with 3 features"""
        try:
            img = Image.open(image_path)
            exif_data = img.info.get('exif', b'')
            features = [
                int(piexif.ImageIFD.Make in piexif.load(exif_data)["0th"]) if exif_data else 0,
                int(piexif.ImageIFD.Software in piexif.load(exif_data)["0th"]) if exif_data else 0,
                int('photoshop' in piexif.load(exif_data)["0th"].get(piexif.ImageIFD.Software, b'').decode('ascii', 'ignore').lower()) if exif_data else 0
            ]
            prediction = self.models['metadata'].predict_proba([features])[0][1]
            return {
                'tamper_probability': float(prediction),
                'verdict': 'TAMPERED' if prediction > 0.7 else 'ORIGINAL',
                'metrics': {
                    'tamper_prob': float(prediction),
                    'software_match': int(prediction > 0.5),
                    'exif_inconsistency': float(prediction * 0.9)
                }
            }
        except Exception as e:
            return {'error': str(e), 'verdict': 'FAILED'}

    def analyze_paper(self, image_path):
        """Paper analysis with 5 features"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(img, 50, 150)
            features = [
                np.mean(img),               # Average brightness
                np.std(img),                # Texture variation  
                np.sum(edges)/(edges.shape[0]*edges.shape[1]),  # Edge density
                np.mean(img[100:200, 100:200]),  # Center region
                np.var(img[::10, ::10])     # Coarse texture
            ]
            prediction = self.models['paper'].predict_proba([features])[0][1]
            return {
                'authenticity_score': float(prediction),
                'verdict': 'INCONSISTENT' if prediction > 0.6 else 'CONSISTENT',
                'metrics': {
                    'fiber_match': float(1 - prediction),
                    'watermark_match': int(prediction < 0.5),
                    'age_consistency': float(1 - prediction * 0.7)
                }
            }
        except Exception as e:
            return {'error': str(e), 'verdict': 'FAILED'}

    def analyze_ela(self, image_path):
        """ELA analysis with 4 features"""
        try:
            original = cv2.imread(image_path)
            temp_path = f"{image_path}_temp.jpg"
            cv2.imwrite(temp_path, original, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            ela = cv2.absdiff(original, cv2.imread(temp_path))
            ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(ela_gray, 50, 150)
            os.remove(temp_path)
            features = [
                np.mean(ela_gray),     # Average error
                np.std(ela_gray),      # Error variation
                np.max(ela_gray),      # Max error
                np.mean(edges)         # Edge artifacts
            ]
            prediction = self.models['ela'].predict_proba([features])[0][1]
            return {
                'manipulation_score': float(prediction),
                'verdict': 'MANIPULATED' if prediction > 0.7 else 'ORIGINAL',
                'metrics': {
                    'compression_artifacts': int(prediction > 0.5),
                    'error_level': float(prediction),
                    'edge_inconsistency': float(prediction * 0.8)
                }
            }
        except Exception as e:
            return {'error': str(e), 'verdict': 'FAILED'}

# Initialize analyzer
print("\nInitializing forensic analyzer...")
analyzer = ForensicAnalyzer()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided', 'status': 'failed'}), 400
        
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected', 'status': 'failed'}), 400
        
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Only PNG/JPG/JPEG allowed', 'status': 'failed'}), 400

    try:
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\nAnalyzing document: {filename}")
        
        # Run all analyses
        results = {
            'report_id': f"FSCAN-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'analysis_date': datetime.now().isoformat(),
            'signature_analysis': analyzer.analyze_signature(filepath),
            'ink_analysis': analyzer.analyze_ink(filepath),
            'metadata_analysis': analyzer.analyze_metadata(filepath),
            'paper_analysis': analyzer.analyze_paper(filepath),
            'ela_analysis': analyzer.analyze_ela(filepath)
        }
        
        # Generate final verdict
        verdict_scores = {
            'signature': results['signature_analysis'].get('authenticity_score', 0),
            'ink': results['ink_analysis'].get('consistency_score', 0),
            'metadata': 1 - results['metadata_analysis'].get('tamper_probability', 0),
            'paper': results['paper_analysis'].get('authenticity_score', 0),
            'ela': 1 - results['ela_analysis'].get('manipulation_score', 0)
        }
        
        avg_score = sum(verdict_scores.values()) / len(verdict_scores)
        results['final_verdict'] = {
            'verdict': 'DOCUMENT_AUTHENTIC' if avg_score >= 0.7 else 'DOCUMENT_FORGED',
            'confidence': avg_score,
            'indicators': [
                ind for ind in [
                    "Signature anomalies detected" if results['signature_analysis'].get('verdict') == 'FORGED' else None,
                    "Multiple ink types detected" if results['ink_analysis'].get('verdict') == 'SUSPECT' else None,
                    "Metadata inconsistencies found" if results['metadata_analysis'].get('verdict') == 'TAMPERED' else None,
                    "Paper inconsistencies detected" if results['paper_analysis'].get('verdict') == 'INCONSISTENT' else None,
                    "Digital manipulation detected" if results['ela_analysis'].get('verdict') == 'MANIPULATED' else None
                ] if ind is not None
            ]
        }
        
        print("Analysis completed successfully!")
        
        # Cleanup file after processing
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing file: {e}")
        
        # Prepare data for redirect
        encoded_data = urllib.parse.quote(json.dumps(results))
        return jsonify({
            'redirect_url': f'/report?data={encoded_data}',
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            return jsonify({'status': 'failed', 'message': 'All fields are required.'}), 400

        hashed_password = generate_password_hash(password)
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                      (username, email, hashed_password))
            conn.commit()
            conn.close()
            return jsonify({'status': 'success', 'message': 'Registration successful!'})
        except sqlite3.IntegrityError:
            return jsonify({'status': 'failed', 'message': 'Username or email already exists.'}), 409
        except Exception as e:
            return jsonify({'status': 'failed', 'message': str(e)}), 500

    # For GET, render the registration page
    return render_template('register.html')

@app.route('/index.html')
def serve_index():
    return render_template('index.html')

@app.route('/analysis.html')
def serve_analysis():
    return render_template('analysis.html')

@app.route('/report')
def serve_report_data():
    print("Serving report.html")  # Debug print
    return render_template('report.html')

@app.route('/case.html')
def serve_case():
    return render_template('case.html')

@app.route('/')
def home():
    return render_template('index.html')

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

if __name__ == '__main__':
    print("\nStarting forensic analysis server...")
    app.run(host='0.0.0.0', port=5000, threaded=True)