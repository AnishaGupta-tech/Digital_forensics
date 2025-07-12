import cv2
import numpy as np
import os
import random

def generate_signature(name, is_forged=False):
    # Create blank white image
    img = np.zeros((150, 300, 3), dtype=np.uint8) + 255
    
    # Generate random signature-like curves
    points = []
    start_x, start_y = 30, 75
    points.append((start_x, start_y))
    
    for i in range(1, 5):
        x = start_x + i * 50 + random.randint(-10, 10)
        y = start_y + random.randint(-20, 20)
        points.append((x, y))
    
    # Draw smooth curve
    for i in range(len(points)-1):
        cv2.line(img, points[i], points[i+1], (0,0,0), random.choice([1,2,2,3]))
    
    # Add flourishes
    if random.random() > 0.3:
        cv2.line(img, points[-1], 
                (points[-1][0]+20, points[-1][1]+10), 
                (0,0,0), 1)
    
    # Add forgery characteristics
    if is_forged:
        # Add tremors
        for i in range(len(points)-1):
            for j in range(3):
                offset_x = random.randint(-2, 2)
                offset_y = random.randint(-2, 2)
                cv2.line(img, 
                        (points[i][0]+offset_x, points[i][1]+offset_y),
                        (points[i+1][0]+offset_x, points[i+1][1]+offset_y),
                        (0,0,0), 1)
        
        # Add extra dots
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, 299)
            y = random.randint(0, 149)
            cv2.circle(img, (x,y), random.randint(1,2), (0,0,0), -1)
    
    # Convert to grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def create_dataset():
    # Create directories
    os.makedirs('dataset/train/genuine', exist_ok=True)
    os.makedirs('dataset/train/forged', exist_ok=True)
    os.makedirs('dataset/test/genuine', exist_ok=True)
    os.makedirs('dataset/test/forged', exist_ok=True)

    # Generate training data (40 samples)
    print("Generating training data...")
    for i in range(20):
        # Genuine
        cv2.imwrite(f'dataset/train/genuine/sig_{i}.png', 
                   generate_signature(f"sig_{i}"))
        # Forged
        cv2.imwrite(f'dataset/train/forged/sig_{i}.png',
                   generate_signature(f"sig_{i}", True))

    # Generate test data (10 samples)
    print("Generating test data...")
    for i in range(20, 25):
        # Genuine test
        cv2.imwrite(f'dataset/test/genuine/test_{i}.png',
                   generate_signature(f"test_{i}"))
        # Forged test
        cv2.imwrite(f'dataset/test/forged/test_{i}.png',
                   generate_signature(f"test_{i}", True))

    print("Dataset generation complete!")
    print(f"Total samples: {20*2} train, {5*2} test")

if __name__ == "__main__":
    create_dataset()