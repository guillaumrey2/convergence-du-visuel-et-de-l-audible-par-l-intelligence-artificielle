import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/set/path/to/color/dataset"  # Updated to RGB dataset
data = pd.read_csv(file_path)

# Preprocess the data
X = data[['red', 'green', 'blue']]  # Use RGB values instead of HSV
y = data['label']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Random Forest Classifier')

# Save the confusion matrix as an image
plt.savefig('confusion_matrix_rf.png', bbox_inches='tight')

# Show the confusion matrix
plt.show()

# Save the models and the label encoder
joblib.dump(rf_model, 'rf_color_classifier_model_rgb.pkl')
joblib.dump(label_encoder, 'label_encoder_rgb.pkl')
print('Models and label encoder saved successfully.')

# Function to predict color label for a new RGB sample using the Random Forest model
def predict_color_rf(rgb_sample):
    # Load the saved model and encoder
    loaded_model = joblib.load('rf_color_classifier_model_rgb.pkl')
    loaded_encoder = joblib.load('label_encoder_rgb.pkl')
    
    # Predict the label
    prediction = loaded_model.predict([rgb_sample])
    color_label = loaded_encoder.inverse_transform(prediction)
    
    return color_label[0]

# Example usage with RGB samples
samples_to_test = [
    [116, 114, 47],   # Example 1
    [50, 88, 95],     # Example 2
    [50, 69, 95],     # Example 3
    [127, 137, 149],  # Example 4
    [141, 146, 146]   # Example 5
]

for sample in samples_to_test:
    predicted_color = predict_color_rf(sample)
    print(f'The predicted color for RGB values {sample} using Random Forest is {predicted_color}')
