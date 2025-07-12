import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm

def generate_ink_dataset(output_dir='ink_dataset', num_samples=500):
    """Generate synthetic documents with different ink characteristics"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/real', exist_ok=True)
    os.makedirs(f'{output_dir}/fake', exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating ink samples"):
        # Create blank document
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255  # White background
        
        # Generate random text with varying ink properties
        font = np.random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX])
        font_scale = np.random.uniform(0.5, 2)
        thickness = np.random.randint(1, 3)
        
        # Real document - consistent ink
        if i < num_samples//2:
            color = (np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(0, 50))
            for j in range(10):  # Add multiple lines
                cv2.putText(img, f"Sample text line {j}", 
                           (50, 100 + j*50), font, font_scale, color, thickness)
            cv2.imwrite(f'{output_dir}/real/doc_{i}.png', img)
        
        # Fake document - inconsistent ink
        else:
            for j in range(10):
                color = (np.random.randint(0, 100), np.random.randint(0, 100), np.random.randint(0, 100))
                cv2.putText(img, f"Sample text line {j}", 
                           (50, 100 + j*50), font, font_scale, color, thickness)
            cv2.imwrite(f'{output_dir}/fake/doc_{i}.png', img)

def extract_ink_features(image_path):
    """Extract ink-related features from an image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for density analysis
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate ink density statistics
    density = 255 - gray  # Invert so higher values = more ink
    density_mean = np.mean(density)
    density_std = np.std(density)
    
    # Color variation analysis
    colored_pixels = img[np.any(img < 240, axis=2)]  # Get non-white pixels
    if len(colored_pixels) > 0:
        color_std = np.std(colored_pixels, axis=0)
    else:
        color_std = np.array([0, 0, 0])
    
    return [density_mean, density_std, *color_std]

def train_ink_model(dataset_dir='ink_dataset'):
    """Train a model to detect ink inconsistencies"""
    X = []
    y = []
    
    # Process real documents
    real_dir = f'{dataset_dir}/real'
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            features = extract_ink_features(os.path.join(real_dir, filename))
            X.append(features)
            y.append(0)  # 0 for real
    
    # Process fake documents
    fake_dir = f'{dataset_dir}/fake'
    for filename in os.listdir(fake_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            features = extract_ink_features(os.path.join(fake_dir, filename))
            X.append(features)
            y.append(1)  # 1 for fake
    
    # Train model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, 'ink_analysis_model.pkl')
    return model

# Generate dataset and train model
generate_ink_dataset()
model = train_ink_model()