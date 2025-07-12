import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score

def generate_paper_dataset(output_dir='paper_dataset', num_samples=500):
    """Generate documents with different paper characteristics"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/real', exist_ok=True)
    os.makedirs(f'{output_dir}/fake', exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating paper samples"):
        # Base paper texture
        if i < num_samples//2:
            # Real document - consistent paper texture
            paper = generate_paper_texture(800, 600, consistency=0.9)
        else:
            # Fake document - inconsistent paper texture
            paper = generate_paper_texture(800, 600, consistency=0.5)
        
        # Add some text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(paper, "Sample Document", (50, 300), font, 1, (0, 0, 0), 2)
        
        # Save
        if i < num_samples//2:
            cv2.imwrite(f'{output_dir}/real/doc_{i}.png', paper)
        else:
            cv2.imwrite(f'{output_dir}/fake/doc_{i}.png', paper)

def generate_paper_texture(height, width, consistency=0.9):
    """Generate paper-like texture"""
    # Base noise
    texture = np.random.normal(loc=230, scale=10, size=(height, width)).clip(200, 255)
    
    # Add fibers (more consistent for real documents)
    for _ in range(int(50 * consistency)):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        length = np.random.randint(10, 50)
        angle = np.random.uniform(0, np.pi)
        
        x2 = int(x + length * np.cos(angle))
        y2 = int(y + length * np.sin(angle))
        
        color = np.random.normal(200, 10)
        cv2.line(texture, (x, y), (x2, y2), color, 1)
    
    # Add some random spots
    for _ in range(int(20 * (1 - consistency))):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        radius = np.random.randint(2, 10)
        color = np.random.normal(200, 20)
        cv2.circle(texture, (x, y), radius, color, -1)
    
    return texture.astype(np.uint8)

def extract_paper_features(image_path):
    """Extract paper texture features"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate texture statistics
    mean_val = np.mean(img)
    std_val = np.std(img)
    
    # Edge detection for fiber analysis
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    # GLCM texture features (simplified)
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return [mean_val, std_val, edge_density, contrast, homogeneity]

def train_paper_model(dataset_dir='paper_dataset'):
    """Train a model to detect paper inconsistencies"""
    X = []
    y = []
    
    # Process real documents
    real_dir = f'{dataset_dir}/real'
    for filename in os.listdir(real_dir):
        if filename.lower().endswith('.png'):
            features = extract_paper_features(os.path.join(real_dir, filename))
            X.append(features)
            y.append(0)  # 0 for real
    
    # Process fake documents
    fake_dir = f'{dataset_dir}/fake'
    for filename in os.listdir(fake_dir):
        if filename.lower().endswith('.png'):
            features = extract_paper_features(os.path.join(fake_dir, filename))
            X.append(features)
            y.append(1)  # 1 for fake
    
    # Train model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Paper Model accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, 'paper_model.pkl')
    return model

# Generate dataset and train model
generate_paper_dataset()
paper_model = train_paper_model()