# Import necessary libraries
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from skimage.segmentation import felzenszwalb
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to apply data augmentation on an image
def augment_image(image):
    print("Applying data augmentation...")
    if np.random.rand() > 0.5:
        # Randomly rotate the image
        angle = np.random.randint(-15, 15)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    if np.random.rand() > 0.5:
        # Randomly flip the image horizontally
        image = cv2.flip(image, 1)
    if np.random.rand() > 0.5:
        # Randomly flip the image vertically
        image = cv2.flip(image, 0)
    if np.random.rand() > 0.5:
        # Randomly adjust HSV values
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:,:,1] *= 0.8 + 0.4 * np.random.rand()
        hsv[:,:,2] *= 0.8 + 0.4 * np.random.rand()
        hsv[:,:,1:3] = np.clip(hsv[:,:,1:3], 0, 255)
        image = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
    if np.random.rand() > 0.5:
        # Add Gaussian noise to the image
        gauss = np.random.normal(0, 0.1**0.5, image.shape)
        image = cv2.add(image, gauss, dtype=cv2.CV_8UC3)
    if np.random.rand() > 0.5:
        # Randomly occlude part of the image
        top = np.random.randint(0, image.shape[0])
        left = np.random.randint(0, image.shape[1])
        bottom = np.random.randint(top, image.shape[0])
        right = np.random.randint(left, image.shape[1])
        image[top:bottom, left:right, :] = 0
    return image

# Function to extract RGB values from images in a dataset
def extract_rgb_values(dataset_path, sample_size=500000):
    print("Extracting RGB values from images...")
    all_rgb_values = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                image = augment_image(image)
                all_rgb_values.append(image.reshape(-1, 3))
            else:
                print(f"Warning: Image {filename} could not be loaded.")
    all_rgb_values = np.vstack(all_rgb_values) if all_rgb_values else np.empty((0, 3))
    if all_rgb_values.shape[0] > sample_size:
        indices = np.random.choice(all_rgb_values.shape[0], sample_size, replace=False)
        all_rgb_values = all_rgb_values[indices]
    return all_rgb_values

# Function to extract a color palette using K-means clustering
def extract_color_palette(rgb_values, n_clusters=180):
    print("Starting K-means clustering...")
    if not rgb_values.size:
        return np.empty((0, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(rgb_values)
    print("K-means clustering completed.")
    return kmeans.cluster_centers_

# Function to repaint an image using a given color palette
def repaint_image_with_palette(image, palette):
    print("Repainting image using the extracted color palette...")
    if not palette.size:
        return image
    repainted_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            closest_color = min(palette, key=lambda x: np.linalg.norm(x - pixel))
            repainted_image[i, j] = closest_color
    return repainted_image

# Function to segment an image using the Felzenszwalb method
def segment_image(image):
    print("Segmenting image...")
    segments = felzenszwalb(image, scale=100, sigma=0.8, min_size=50)
    print("Segmentation completed.")
    return segments

# Function to extract adjacent color features from segmented image
def extract_adjacent_color_features(image, segments):
    print("Extracting adjacent color features...")
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
    return feature_vectors.flatten() if feature_vectors.size else np.zeros(feature_vector_size * num_segments)

# Function to load annotations from an Excel file
def load_annotations(annotations_path):
    print("Loading annotations...")
    annotations = pd.read_excel(annotations_path, engine='odf')
    annotations['LABEL'] = annotations['LABEL'].apply(lambda x: 1 if x == 'positive' else 0)
    print(f"Loaded {len(annotations)} annotations.")
    return annotations

# Function to plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

# Function to plot an ROC curve
def plot_roc_curve(y_true, y_scores, model_name="Model"):
    print(f"Plotting ROC curve for {model_name}...")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Define paths for the dataset and annotations
dataset_path = "/set/path/to/images/folder"
annotations_path = "/set/path/to/annotation/file.ods"

# Extract RGB values from the dataset
rgb_values = extract_rgb_values(dataset_path)

# Extract color palette using K-means clustering
color_palette = extract_color_palette(rgb_values)

# Save the color palette to a file
np.save('color_palette.npy', color_palette)

# Load annotations
annotations = load_annotations(annotations_path)

# Initialize lists for features and labels
features_list, labels_list = [], []

# Process each image according to the annotations
for index, row in annotations.iterrows():
    image_path = os.path.join(dataset_path, row['CODE'] + '.jpg')
    image = cv2.imread(image_path)
    if image is not None:
        image = augment_image(image)
        print(f"Processing image: {image_path}")
        repainted_image = repaint_image_with_palette(image, color_palette)
        segments = segment_image(repainted_image)
        features = extract_adjacent_color_features(repainted_image, segments)
        if features.size:
            features_list.append(features)
            labels_list.append(row['LABEL'])
        else:
            print(f"No features extracted for {image_path}")
    else:
        print(f"Image {image_path} could not be loaded.")

# Pad feature vectors to have the same length
max_features_length = max(len(f) for f in features_list)
features_list = [np.pad(f, (0, max_features_length - len(f)), 'constant') for f in features_list]

# Convert lists to numpy arrays
X = np.array(features_list)
y = np.array(labels_list)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Predict the labels for the test set
y_pred_tree = tree.predict(X_test)
y_scores_tree = tree.predict_proba(X_test)[:, 1]

# Print Decision Tree results
print("Decision Tree Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_tree)}')
print('Classification Report:')
print(classification_report(y_test, y_pred_tree))

# Plot and display confusion matrix
plot_confusion_matrix(y_test, y_pred_tree, title='Decision Tree Confusion Matrix')

# Plot and display ROC curve
plot_roc_curve(y_test, y_scores_tree, model_name="Decision Tree")

# Save the trained model to a file
joblib.dump(tree, 'emotion_classifier_model.pkl')