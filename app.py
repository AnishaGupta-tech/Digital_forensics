import os
import tempfile
import pdf2image
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import pytesseract
import exifread
from sklearn.ensemble import RandomForestClassifier
import joblib
import hashlib
from pdfminer.high_level import extract_text

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models (in a real app, these would be properly trained)
signature_model = joblib.load('signature_model.pkl')  # Placeholder
text_tamper_model = joblib.load('text_tamper_model.pkl')  # Placeholder

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_metadata(filepath):
    """Extract metadata from PDF or image"""
    metadata = {}
    
    if filepath.lower().endswith('.pdf'):
        try:
            text = extract_text(filepath)
            metadata['text_content'] = text[:500]  # First 500 chars
        except:
            metadata['text_content'] = "Could not extract text"
    else:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f)
            metadata['exif_data'] = {tag: str(tags[tag]) for tag in tags}
    
    # Add file hash
    with open(filepath, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    metadata['file_hash'] = file_hash
    
    return metadata

def analyze_signature(image):
    """Analyze signature using computer vision"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Feature extraction (simplified for example)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    # Predict using model (in a real app, this would be more sophisticated)
    features = np.array([[edge_density, 0.5, 0.3]])  # Placeholder features
    is_forged = signature_model.predict(features)[0]
    confidence = signature_model.predict_proba(features)[0][1]
    
    return {
        'is_forged': bool(is_forged),
        'confidence': float(confidence),
        'analysis': {
            'edge_density': edge_density,
            'stroke_consistency': 0.85 if not is_forged else 0.15
        }
    }

def analyze_text_tampering(image):
    """Detect text tampering"""
    # OCR the text
    text = pytesseract.image_to_string(image)
    
    # Analyze text (simplified example)
    features = np.array([[len(text), 0.7, 0.2]])  # Placeholder features
    is_tampered = text_tamper_model.predict(features)[0]
    confidence = text_tamper_model.predict_proba(features)[0][1]
    
    return {
        'is_tampered': bool(is_tampered),
        'confidence': float(confidence),
        'text_analysis': {
            'ocr_text': text[:200],  # First 200 chars
            'ink_consistency': 0.92 if not is_tampered else 0.18
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        results = {
            'filename': filename,
            'analysis': {}
        }
        
        try:
            # Extract metadata
            results['metadata'] = extract_metadata(filepath)
            
            # Convert PDF to images if needed
            if filename.lower().endswith('.pdf'):
                images = pdf2image.convert_from_path(filepath)
                image = images[0]  # Analyze first page
            else:
                image = Image.open(filepath)
            
            # Analyze signature
            results['analysis']['signature'] = analyze_signature(image)
            
            # Analyze text
            results['analysis']['text'] = analyze_text_tampering(image)
            
            # Overall verdict
            sig_conf = results['analysis']['signature']['confidence']
            text_conf = results['analysis']['text']['confidence']
            overall_conf = (sig_conf + text_conf) / 2
            
            results['verdict'] = {
                'is_forged': overall_conf > 0.7,
                'confidence': overall_conf,
                'message': 'Likely forged' if overall_conf > 0.7 else 'Likely authentic'
            }
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)