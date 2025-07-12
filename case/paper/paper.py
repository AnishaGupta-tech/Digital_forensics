import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_paper(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast for paper texture analysis
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Edge detection for paper fibers
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Texture analysis using GLCM (simplified)
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(enhanced, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Watermark detection (basic approach)
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    watermark = cv2.subtract(blurred, gray)
    _, watermark_thresh = cv2.threshold(watermark, 10, 255, cv2.THRESH_BINARY)
    
    # Display results
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Document')
    
    plt.subplot(2, 3, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Contrast')
    
    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection (Paper Fibers)')
    
    plt.subplot(2, 3, 4)
    plt.imshow(watermark, cmap='gray')
    plt.title('Potential Watermark')
    
    plt.subplot(2, 3, 5)
    plt.imshow(watermark_thresh, cmap='gray')
    plt.title('Thresholded Watermark')
    
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.6, f'Texture Analysis:\nContrast: {contrast:.2f}\nHomogeneity: {homogeneity:.2f}', 
             fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'watermark_present': np.sum(watermark_thresh) > 1000
    }

# Usage
paper_analysis = analyze_paper('document.jpg')
print("Paper Analysis Results:", paper_analysis)