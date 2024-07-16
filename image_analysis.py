import os
import sys
import cv2
import numpy as np
import json
import joblib
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.segmentation import felzenszwalb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Set the model path relative to the script location
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

# Load trained models and resources
color_model = joblib.load(os.path.join(MODEL_PATH, 'color_classifier_model.pkl'))
color_encoder = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))
emotion_model = joblib.load(os.path.join(MODEL_PATH, 'emotion_classifier_model.pkl'))
color_palette = np.load(os.path.join(MODEL_PATH, 'color_palette.npy'))
art_style_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'art_style_model.h5'))

# Normalization ranges based on your data analysis
COARSENESS_RANGE = (0, 5)
CONTRAST_RANGE = (0, 100)
DIRECTIONALITY_RANGE = (0, 0.5)
ROUGHNESS_RANGE = (0, 100)
REGULARITY_RANGE = (0, 30)

# Define reverse mapping dictionary for style prediction
style_map_reverse = {
    0: 'abstract',
    1: 'color_field',
    2: 'cubism',
    3: 'expressionism',
    4: 'impressionism',
    5: 'realism',
    6: 'romanticism',
    7: 'symbolism'
}

def tamura_preprocess_image(image):
    print("Preprocessing the image for grayscale and blur...")
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (256, 256))
    smooth_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
    return smooth_img

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

def contrast(image):
    image = np.array(image, dtype=np.float32).flatten()
    m4 = np.mean((image - np.mean(image))**4)
    v = np.var(image)
    std = np.sqrt(v)
    
    if v == 0:
        alfa4 = 0  # Handle division by zero
    else:
        alfa4 = m4 / (v**2)
    
    if alfa4 == 0:
        fcon = 0  # Handle division by zero
    else:
        fcon = std / (np.power(alfa4, 0.25))
    
    return fcon

def directionality(image):
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) * (180 / np.pi) % 180
    histogram, bins = np.histogram(orientation, bins=36, range=(0, 180), weights=magnitude)
    
    max_index = np.argmax(histogram)
    main_direction = bins[max_index] if histogram.size > 0 else 0
    direction_strength = histogram[max_index] / np.sum(histogram) if np.sum(histogram) != 0 else 0
    
    return direction_strength, main_direction

def regularity(image, block_size=32):
    scores = []
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j+j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                scores.append((coarseness(block, 5), contrast(block)))
    scores = np.array(scores)
    std_devs = np.std(scores, axis=0)
    return np.mean(std_devs)

def roughness(fcrs, fcon):
    return fcrs + fcon

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def calculate_tamura_features(image):
    print("Calculating Tamura features")
    fcrs = coarseness(image, 5)
    fcon = contrast(image)
    fd, main_direction = directionality(image)
    freg = regularity(image)
    f_rgh = roughness(fcrs, fcon)
    
    # Normalize features
    fcrs_normalized = normalize(fcrs, COARSENESS_RANGE[0], COARSENESS_RANGE[1])
    fcon_normalized = normalize(fcon, CONTRAST_RANGE[0], CONTRAST_RANGE[1])
    fd_normalized = normalize(fd, DIRECTIONALITY_RANGE[0], DIRECTIONALITY_RANGE[1])
    freg_normalized = normalize(freg, REGULARITY_RANGE[0], REGULARITY_RANGE[1])
    f_rgh_normalized = normalize(f_rgh, ROUGHNESS_RANGE[0], ROUGHNESS_RANGE[1])
    
    return fcrs_normalized, fcon_normalized, fd_normalized, main_direction, freg_normalized, f_rgh_normalized

def color_preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred_image = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    resized_image = cv2.resize(blurred_image, (100, 100))
    return resized_image

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

def filter_dark_pixels(pixel_values, dark_threshold=30, black_threshold=0.5):
    dark_pixels = np.all(pixel_values < [dark_threshold, dark_threshold, dark_threshold], axis=1)
    dark_proportion = np.sum(dark_pixels) / pixel_values.shape[0]
    print(f"Proportion of dark pixels: {dark_proportion}")

    if (dark_proportion < black_threshold):
        pixel_values = pixel_values[~dark_pixels]
    
    return pixel_values

def find_dominant_color(image, max_k=10, dark_threshold=30, black_threshold=0.5):
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

def predict_color_label(dominant_color):
    print("Predicting color label")
    dominant_color_rgb = np.uint8([dominant_color]).reshape(1, -1)
    print(f"RGB value sent for prediction: {dominant_color_rgb}")
    prediction = color_model.predict(dominant_color_rgb)
    color_label = color_encoder.inverse_transform(prediction)
    return color_label[0]

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

def segment_image(image):
    print("Segmenting image...")
    segments = felzenszwalb(image, scale=100, sigma=0.8, min_size=50)
    return segments

def extract_features(image, segments, palette):
    print("Extracting features...")
    h, w = image.shape[:2]
    num_segments = np.max(segments) + 1
    features = []
    for seg_id in range(num_segments):
        mask = (segments == seg_id)
        segment_color = np.mean(image[mask], axis=0)
        segment_size = np.sum(mask) / (h * w)
        center_of_mass = np.array(np.nonzero(mask)).mean(axis=1) / np.array([h, w])
        adjacent_colors = []
        for i in range(h):
            for j in range(w):
                if segments[i, j] == seg_id:
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and segments[ni, nj] != seg_id:
                            adjacent_colors.append(image[ni, nj])
        adjacent_color = np.mean(adjacent_colors, axis=0) if adjacent_colors else segment_color
        features.append(np.concatenate([segment_color, [segment_size], center_of_mass, adjacent_color]))
    features = np.array(features).flatten()
    
    expected_features = emotion_model.n_features_in_
    if len(features) < expected_features:
        features = np.pad(features, (0, expected_features - len(features)), 'constant')
    elif len(features) > expected_features:
        features = features[:expected_features]
    
    return features

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
    
    processed_image = tamura_preprocess_image(image)
    fcrs, fcon, fd, main_direction, freg, f_rgh = calculate_tamura_features(processed_image)
    
    dominant_color = find_dominant_color(image)
    color_label = predict_color_label(dominant_color)
    
    repainted_image = repaint_image_with_palette(image, color_palette)
    segments = segment_image(repainted_image)
    features = extract_features(repainted_image, segments, color_palette)
    features = features.reshape(1, -1)

    predicted_emotion = emotion_model.predict(features)
    emotion_label = 'positive' if predicted_emotion == 1 else 'negative'
    
    keras_image = load_img(image_path, target_size=(224, 224))
    keras_image = img_to_array(keras_image)
    keras_image = np.expand_dims(keras_image, axis=0)
    keras_image /= 255.0
    style_prediction = art_style_model.predict(keras_image)
    style_label_index = np.argmax(style_prediction)
    style_label = style_map_reverse[style_label_index]
    
    result = {
        "image": base_filename,
        "dominant_color": color_label,
        "emotion": emotion_label,
        "art_style": style_label,
        "texture": {
            "coarseness": fcrs,
            "contrast": fcon,
            "directionality": fd,
            "main_direction": main_direction,
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python full_pipeline.py <image_path> <output_dir>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    analyze_image(image_path, output_dir)
