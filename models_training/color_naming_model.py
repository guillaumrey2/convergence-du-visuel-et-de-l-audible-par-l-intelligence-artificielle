# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset from the specified file path
file_path = "/set/path/to/color/dataset"
data = pd.read_csv(file_path)

# Define feature variables (RGB values) and target variable (color label)
X = data[['red', 'green', 'blue']]
y = data['label']

# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Print a detailed classification report
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# Generate and visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Random Forest Classifier')

# Save the confusion matrix plot as an image file
plt.savefig('confusion_matrix_rf.png', bbox_inches='tight')

# Display the confusion matrix plot
plt.show()

# Save the trained model and label encoder to disk
joblib.dump(rf_model, 'rf_color_classifier_model_rgb.pkl')
joblib.dump(label_encoder, 'label_encoder_rgb.pkl')
print('Models and label encoder saved successfully.')

# Define a function to predict the color label for a given RGB sample using the trained model
def predict_color_rf(rgb_sample):
    # Load the trained model and label encoder from disk
    loaded_model = joblib.load('rf_color_classifier_model_rgb.pkl')
    loaded_encoder = joblib.load('label_encoder_rgb.pkl')
    
    # Predict the label for the given RGB sample
    prediction = loaded_model.predict([rgb_sample])
    
    # Convert the predicted label back to the original color label
    color_label = loaded_encoder.inverse_transform(prediction)
    
    # Return the color label
    return color_label[0]
