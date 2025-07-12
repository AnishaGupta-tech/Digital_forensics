import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def analyze_ink(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 3)
    
    # Perform K-means clustering to find dominant ink colors
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)
    
    # Get the dominant colors
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Count the pixels in each cluster
    counts = np.bincount(labels)
    
    # Plot the dominant colors
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Document')
    
    plt.subplot(1, 2, 2)
    plt.pie(counts, colors=colors/255, labels=[f'Ink {i+1}' for i in range(len(colors))])
    plt.title('Dominant Ink Colors')
    
    plt.show()
    
    # Calculate ink density variations
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    density_map = cv2.GaussianBlur(gray, (15, 15), 0)
    
    plt.imshow(density_map, cmap='jet')
    plt.title('Ink Density Variations')
    plt.colorbar()
    plt.show()
    
    return colors, density_map

# Usage
ink_colors, density_map = analyze_ink('document.jpg')