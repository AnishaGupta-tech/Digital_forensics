import cv2
import numpy as np
import matplotlib.pyplot as plt

def perform_ela(image_path, quality=90):
    # Read the original image
    original = cv2.imread(image_path)
    
    # Save the image with specified quality then read it back
    temp_path = 'temp_ela.jpg'
    cv2.imwrite(temp_path, original, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed = cv2.imread(temp_path)
    
    # Calculate the difference
    ela_image = cv2.absdiff(original, compressed)
    
    # Scale the differences for better visualization
    ela_image = ela_image.astype(np.float32) / 255
    ela_image = (ela_image * 255).astype(np.uint8)
    
    # Convert to grayscale for analysis
    ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate ELA score (higher values indicate more manipulation)
    ela_score = np.mean(ela_gray)
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(ela_image, cv2.COLOR_BGR2RGB))
    plt.title('ELA Image')
    
    plt.subplot(1, 3, 3)
    plt.imshow(ela_gray, cmap='jet')
    plt.title(f'ELA Heatmap (Score: {ela_score:.2f})')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return ela_image, ela_score

# Usage
ela_result, ela_score = perform_ela('document.jpg')