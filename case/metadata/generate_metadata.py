import os
from PIL import Image
import piexif
from datetime import datetime, timedelta
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def sanitize_exif_dict(exif_dict):
    """Sanitize EXIF dictionary to avoid None values which cause piexif errors"""
    for ifd in exif_dict:
        if isinstance(exif_dict[ifd], dict):
            for tag in list(exif_dict[ifd].keys()):
                value = exif_dict[ifd][tag]
                if value is None:
                    if isinstance(tag, int):
                        # Assign dummy byte or string based on field
                        exif_dict[ifd][tag] = b"Unknown" if isinstance(value, bytes) else "Unknown"
    return exif_dict

def generate_metadata_dataset(output_dir='metadata_dataset', num_samples=500):
    """Generate images with different metadata patterns"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/real', exist_ok=True)
    os.makedirs(f'{output_dir}/fake', exist_ok=True)

    # Create a simple image to use as base
    img = Image.new('RGB', (600, 800), color='white')

    for i in range(num_samples):
        if i < num_samples // 2:
            # Real document - consistent metadata
            exif_dict = {
                "0th": {
                    piexif.ImageIFD.Make: b"Canon",
                    piexif.ImageIFD.Model: b"Canon EOS 5D",
                    piexif.ImageIFD.Software: b"Scanner v1.0"
                },
                "Exif": {
                    piexif.ExifIFD.DateTimeOriginal: datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    piexif.ExifIFD.ExifVersion: b"0230"
                },
                "1st": {},
                "GPS": {},
                "Interop": {},
                "thumbnail": None
            }
            output_path = f'{output_dir}/real/doc_{i}.jpg'
        else:
            # Fake document - inconsistent metadata
            exif_dict = {
                "0th": {
                    piexif.ImageIFD.Make: random.choice([b"Canon", b"Nikon", b"Sony", None]),
                    piexif.ImageIFD.Model: random.choice([b"EOS 5D", b"D850", b"a7R IV", b"Unknown"]),
                    piexif.ImageIFD.Software: random.choice([b"Photoshop 2023", b"GIMP", b"Scanner v1.0", None])
                },
                "Exif": {
                    piexif.ExifIFD.DateTimeOriginal: (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y:%m:%d %H:%M:%S"),
                    piexif.ExifIFD.ExifVersion: random.choice([b"0230", b"0220", b"0210", None])
                },
                "1st": {},
                "GPS": {},
                "Interop": {},
                "thumbnail": None
            }
            exif_dict = sanitize_exif_dict(exif_dict)
            output_path = f'{output_dir}/fake/doc_{i}.jpg'

        # Save with metadata
        try:
            exif_bytes = piexif.dump(exif_dict)
            img.save(output_path, exif=exif_bytes, quality=95)
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

def extract_metadata_features(image_path):
    """Extract metadata features from an image"""
    features = {}

    try:
        img = Image.open(image_path)
        exif_data = img.info.get('exif')

        if exif_data:
            exif_dict = piexif.load(exif_data)

            features['has_make'] = int(exif_dict["0th"].get(piexif.ImageIFD.Make) is not None)
            features['has_model'] = int(exif_dict["0th"].get(piexif.ImageIFD.Model) is not None)
            features['has_software'] = int(exif_dict["0th"].get(piexif.ImageIFD.Software) is not None)

            software = exif_dict["0th"].get(piexif.ImageIFD.Software, b'').decode('ascii', errors='ignore').lower()
            features['has_photoshop'] = int('photoshop' in software)
            features['has_gimp'] = int('gimp' in software)

            try:
                date_str = exif_dict["Exif"].get(piexif.ExifIFD.DateTimeOriginal, b'').decode('ascii')
                features['has_date'] = int(bool(date_str))
            except:
                features['has_date'] = 0
        else:
            features = {key: 0 for key in [
                'has_make', 'has_model', 'has_software',
                'has_photoshop', 'has_gimp', 'has_date'
            ]}
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        features = {key: 0 for key in [
            'has_make', 'has_model', 'has_software',
            'has_photoshop', 'has_gimp', 'has_date'
        ]}
    
    return features

def train_metadata_model(dataset_dir='metadata_dataset'):
    """Train a model to detect metadata inconsistencies"""
    X, y = [], []

    for label, subfolder in enumerate(['real', 'fake']):
        folder_path = os.path.join(dataset_dir, subfolder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.jpg'):
                path = os.path.join(folder_path, filename)
                features = extract_metadata_features(path)
                X.append(list(features.values()))
                y.append(label)  # 0 = real, 1 = fake

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Metadata Model Accuracy: {accuracy:.2%}")

    joblib.dump(model, 'metadata_model.pkl')
    return model

# Run everything
generate_metadata_dataset()
metadata_model = train_metadata_model()
