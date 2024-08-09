# Import necessary libraries
import os
import sys
import cv2
import numpy as np
import json
import joblib
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore # Image processing from Keras
from skimage.segmentation import felzenszwalb  # Image segmentation from skimage
from sklearn.cluster import KMeans  # KMeans clustering from sklearn
from sklearn.metrics import silhouette_score  # Silhouette score for clustering evaluation
import tensorflow as tf  # TensorFlow for deep learning models
import matplotlib.pyplot as plt  # Plotting library

# Set the model path relative to the script location
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

# Load trained models and resources
color_model = joblib.load(os.path.join(MODEL_PATH, 'color_classifier_model.pkl'))
color_encoder = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))
emotion_model = joblib.load(os.path.join(MODEL_PATH, 'emotion_classifier_model.pkl'))
color_palette = np.load(os.path.join(MODEL_PATH, 'color_palette.npy'))
art_style_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'art_style_model.h5'))

# Normalization ranges based on data analysis
COARSENESS_RANGE = (1, 4)
CONTRAST_RANGE = (0, 100)
DIRECTIONALITY_RANGE = (0, 0.5)
ROUGHNESS_RANGE = (0, 100)
REGULARITY_RANGE = (0, 30)

# Define reverse mapping dictionary for style prediction
style_map_reverse = {
    0: 'abstract',
    1: 'impressionism',
    2: 'realism',
    3: 'romanticism',
    4: 'symbolism',
    5: 'color_field',
    6: 'cubism',
    7: 'expressionism'
}

# ----- TEXTURE ----- #

# Function to preprocess the image for texture analysis
def tamura_preprocess_image(image, target_size=(256, 256)):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray_img.shape[:2]
    target_w, target_h = target_size
    aspect_ratio = w / h
    
    if aspect_ratio > 1:  # Width is greater, resize by width
        new_w = target_w
        new_h = int(new_w / aspect_ratio)
    else:  # Height is greater, resize by height
        new_h = target_h
        new_w = int(new_h * aspect_ratio)
    
    resized_img = cv2.resize(gray_img, (new_w, new_h))
    smooth_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
    return smooth_img

# Function to calculate coarseness
def coarseness(image, kmax):
    w, h = image.shape
    kmax = min(kmax, int(np.log2(w)), int(np.log2(h)))
    average_gray = np.zeros([kmax, w, h])
    horizon = np.zeros([kmax, w, h])
    vertical = np.zeros([kmax, w, h])
    Sbest = np.zeros([w, h])

    for k in range(kmax):
        window = np.power(2, k)
        for wi in range(window, w-window):
            for hi in range(window, h-window):
                average_gray[k, wi, hi] = np.mean(image[wi-window:wi+window+1, hi-window:hi+window+1])
        for wi in range(window, w-window):
            for hi in range(window, h-window):
                horizon[k, wi, hi] = average_gray[k, wi+window, hi] - average_gray[k, wi-window, hi]
                vertical[k, wi, hi] = average_gray[k, wi, hi+window] - average_gray[k, wi, hi-window]
        horizon[k] *= (1.0 / (2 * window))
        vertical[k] *= (1.0 / (2 * window))

    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizon[:, wi, hi])
            v_max = np.max(vertical[:, wi, hi])
            Sbest[wi, hi] = np.power(2, np.argmax(horizon[:, wi, hi] >= v_max))

    fcrs = np.mean(Sbest)
    return fcrs

# Function to calculate contrast
def contrast(image):
    image = np.array(image, dtype=np.float32).flatten()
    m4 = np.mean((image - np.mean(image))**4)
    v = np.var(image)
    std = np.sqrt(v)
    
    if v == 0:
        alfa4 = 0
    else:
        alfa4 = m4 / (v**2)
    
    if alfa4 == 0:
        fcon = 0
    else:
        fcon = std / (np.power(alfa4, 0.25))
    
    return fcon

# Function to calculate directionality
def directionality(image):
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) * (180 / np.pi) % 180
    histogram, bins = np.histogram(orientation, bins=36, range=(0, 180), weights=magnitude)
    
    max_index = np.argmax(histogram)
    direction_strength = histogram[max_index] / np.sum(histogram) if np.sum(histogram) != 0 else 0
    
    return direction_strength

# Function to calculate regularity
def regularity(image, block_size=32):
    scores = []
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                scores.append((coarseness(block, 5), contrast(block)))
    if scores:
        scores = np.array(scores)
        std_devs = np.std(scores, axis=0)
        return np.mean(std_devs)
    else:
        return 0

# Function to calculate roughness
def roughness(fcrs, fcon):
    return fcrs + fcon

# Function to normalize a value
def normalize(value, min_val, max_val):
    if value < min_val:
        return 0.0
    elif value > max_val:
        return 1.0
    else:
        return (value - min_val) / (max_val - min_val)

# Function to calculate Tamura texture features
def calculate_tamura_features(image):
    print("Calculating Tamura features")
    fcrs = coarseness(image, 5)
    print(f"Raw coarseness value: {fcrs}")
    fcon = contrast(image)
    fd = directionality(image)
    freg = regularity(image)
    f_rgh = roughness(fcrs, fcon)
    
    # Normalize features
    fcrs_normalized = normalize(fcrs, COARSENESS_RANGE[0], COARSENESS_RANGE[1])
    print(f"Normalized coarseness value: {fcrs_normalized}")
    fcon_normalized = normalize(fcon, CONTRAST_RANGE[0], CONTRAST_RANGE[1])
    fd_normalized = normalize(fd, DIRECTIONALITY_RANGE[0], DIRECTIONALITY_RANGE[1])
    freg_normalized = normalize(freg, REGULARITY_RANGE[0], REGULARITY_RANGE[1])
    f_rgh_normalized = normalize(f_rgh, ROUGHNESS_RANGE[0], ROUGHNESS_RANGE[1])
    
    return fcrs_normalized, fcon_normalized, fd_normalized, freg_normalized, f_rgh_normalized

# ----- COLOR ----- #

# Function to preprocess the image for color analysis
def color_preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred_image = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    resized_image = cv2.resize(blurred_image, (250, 250))
    return resized_image

# Function to find the optimal number of clusters for KMeans
def find_optimal_clusters(pixel_values, max_k=10):
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(pixel_values)
        labels = kmeans.labels_
        score = silhouette_score(pixel_values, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

# Function to filter out dark pixels
def filter_dark_pixels(pixel_values, dark_threshold=30, black_threshold=0.3):
    dark_pixels = np.all(pixel_values < [dark_threshold, dark_threshold, dark_threshold], axis=1)
    dark_proportion = np.sum(dark_pixels) / pixel_values.shape[0]
    print(f"Proportion of dark pixels: {dark_proportion}")

    if (dark_proportion < black_threshold):
        pixel_values = pixel_values[~dark_pixels]
    
    return pixel_values

# Function to find the dominant color in an image
def find_dominant_color(image, max_k=10, dark_threshold=30, black_threshold=0.3):
    print("Finding dominant color")
    image_rgb = color_preprocess_image(image)
    pixel_values = image_rgb.reshape((-1, 3))

    pixel_values = filter_dark_pixels(pixel_values, dark_threshold, black_threshold)
    
    if pixel_values.size == 0:
        return np.array([0, 0, 0])

    optimal_k = find_optimal_clusters(pixel_values, max_k=max_k)
    print(f"Optimal number of clusters: {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(pixel_values)
    dominant_colors = kmeans.cluster_centers_

    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    print("Cluster centers (RGB values):")
    for i, color in enumerate(dominant_colors):
        print(f"Cluster {i}: {color} - Occurrences: {counts[i]}")

    plt.figure(figsize=(8, 2))
    plt.subplot(1, optimal_k + 1, 1)
    plt.imshow([dominant_colors / 255])
    plt.title("Cluster Colors")

    for i, color in enumerate(dominant_colors):
        plt.subplot(1, optimal_k + 1, i + 2)
        plt.imshow([[color / 255]])
        plt.title(f"Cluster {i}")

    plt.show()

    dominant_color = dominant_colors[np.argmax(counts)]
    print(f"Extracted dominant color (RGB): {dominant_color}")
    return dominant_color

# Function to predict the color label
def predict_color_label(dominant_color):
    print("Predicting color label")
    dominant_color_rgb = np.uint8([dominant_color]).reshape(1, -1)
    print(f"RGB value sent for prediction: {dominant_color_rgb}")
    prediction = color_model.predict(dominant_color_rgb)
    color_label = color_encoder.inverse_transform(prediction)
    return color_label[0]

# ----- EMOTION ----- #

# Function to repaint the image with the given color palette
def repaint_image_with_palette(image, palette):
    print("Repainting image")
    h, w, _ = image.shape
    repainted_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel = image[i, j]
            closest_palette_color = min(palette, key=lambda x: np.linalg.norm(x - pixel))
            repainted_image[i, j] = closest_palette_color
    return repainted_image

# Function to segment the image
def segment_image(image):
    print("Segmenting image...")
    segments = felzenszwalb(image, scale=100, sigma=0.8, min_size=50)
    return segments

# Function to extract adjacent color features from the image segments
def extract_adjacent_color_features(image, segments):
    num_segments = np.max(segments) + 1
    feature_vector_size = 10
    feature_vectors = np.zeros((num_segments, feature_vector_size))

    for seg_id in range(num_segments):
        mask = segments == seg_id
        if np.any(mask):
            segment_color = np.mean(image[mask], axis=0)
            segment_area = np.sum(mask)
            feature_vectors[seg_id, :3] = segment_color
            feature_vectors[seg_id, 3] = segment_area
    
    features = feature_vectors.flatten()
    return features

# Main function to analyze the image
def analyze_image(image_path, output_dir):
    print(f"Analyzing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Loaded Image")
    plt.show()
    
    processed_image = tamura_preprocess_image(image,)
    fcrs, fcon, fd, freg, f_rgh = calculate_tamura_features(processed_image)
    
    dominant_color = find_dominant_color(image)
    color_label = predict_color_label(dominant_color)
    
    repainted_image = repaint_image_with_palette(image, color_palette)
    segments = segment_image(repainted_image)
    features = extract_adjacent_color_features(repainted_image, segments)
    expected_feature_length = 39880
    if len(features) < expected_feature_length:
        features = np.pad(features, (0, expected_feature_length - len(features)), 'constant')
    else:
        features = features[:expected_feature_length]
    
    features = features.reshape(1, -1)

    predicted_emotion = emotion_model.predict(features)
    predicted_emotion_label = predicted_emotion[0]
    predicted_emotion_prob = emotion_model.predict_proba(features)[:, predicted_emotion_label]

    emotion_label = 'positive' if predicted_emotion == 1 else 'negative'

    print(f"Predicted Emotion: {emotion_label}")
    print(f"Probability of positive emotion: {predicted_emotion_prob[0]:.2f}")
    
    keras_image = load_img(image_path, target_size=(224, 224))
    keras_image = img_to_array(keras_image)
    keras_image = np.expand_dims(keras_image, axis=0)
    keras_image /= 255.0
    style_prediction = art_style_model.predict(keras_image)
    style_label_index = np.argmax(style_prediction)
    style_label = style_map_reverse[style_label_index]

    style_probabilities = {style_map_reverse[i]: float(prob) * 100 for i, prob in enumerate(style_prediction[0])}

    print(f"Predicted Style: {style_label}")
    print(f"Style Probabilities: {style_probabilities}")
    
    result = {
        "image": base_filename,
        "dominant_color": color_label,
        "emotion": emotion_label,
        "art_style": style_label,
        "texture": {
            "coarseness": fcrs,
            "contrast": fcon,
            "directionality": fd,
            "regularity": freg,
            "roughness": f_rgh
        }
    }
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_filename = os.path.join(output_dir, base_filename + '.json')
    with open(json_filename, 'w') as json_file:
        json.dump(result, json_file, indent=4)
    
    print(f"Analysis results saved to {json_filename}")

# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python full_pipeline.py <image_path> <output_dir>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    analyze_image(image_path, output_dir)