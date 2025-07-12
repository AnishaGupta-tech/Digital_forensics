import cv2
import numpy as np
import os
from tqdm import tqdm

def generate_ela_dataset(output_dir='ela_dataset', num_samples=500):
    """Generate dataset with authentic and manipulated images"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/real', exist_ok=True)
    os.makedirs(f'{output_dir}/fake', exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating ELA samples"):
        # Create random document background
        if np.random.rand() > 0.5:
            # Texture paper background
            bg = np.random.normal(loc=230, scale=10, size=(800, 600, 3)).clip(0, 255).astype(np.uint8)
        else:
            # Clean white background
            bg = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Add some random text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bg, "Sample Document", (50, 300), font, 1, (0, 0, 0), 2)
        
        if i < num_samples//2:
            # Real document - save with consistent quality
            cv2.imwrite(f'{output_dir}/real/doc_{i}.jpg', bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            # Fake document - composite or edited
            # Add a random rectangle (simulating pasted element)
            x, y = np.random.randint(100, 400), np.random.randint(100, 400)
            w, h = np.random.randint(50, 200), np.random.randint(50, 100)
            bg[y:y+h, x:x+w] = np.random.randint(200, 255, (h, w, 3))
            
            # Save with varying quality to simulate editing
            cv2.imwrite(f'{output_dir}/fake/doc_{i}.jpg', bg, [cv2.IMWRITE_JPEG_QUALITY, np.random.randint(70, 90)])

def extract_ela_features(image_path, quality=90):
    """Extract ELA features from an image"""
    original = cv2.imread(image_path)
    temp_path = 'temp_ela.jpg'
    
    # Save and re-load to compute ELA
    cv2.imwrite(temp_path, original, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed = cv2.imread(temp_path)
    
    # Calculate ELA
    ela = cv2.absdiff(original, compressed)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    
    # Extract features
    mean_ela = np.mean(ela_gray)
    std_ela = np.std(ela_gray)
    max_ela = np.max(ela_gray)
    
    # Additional texture features
    edges = cv2.Canny(ela_gray, 50, 150)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    return [mean_ela, std_ela, max_ela, edge_density]

def train_ela_model(dataset_dir='ela_dataset'):
    """Train a model to detect manipulation using ELA features"""
    X = []
    y = []
    
    # Process real documents
    real_dir = f'{dataset_dir}/real'
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            features = extract_ela_features(os.path.join(real_dir, filename))
            X.append(features)
            y.append(0)  # 0 for real
    
    # Process fake documents
    fake_dir = f'{dataset_dir}/fake'
    for filename in os.listdir(fake_dir):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            features = extract_ela_features(os.path.join(fake_dir, filename))
            X.append(features)
            y.append(1)  # 1 for fake
    
    # Train model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"ELA Model accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, 'ela_model.pkl')
    return model

# Generate dataset and train model
generate_ela_dataset()
ela_model = train_ela_model()