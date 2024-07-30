# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16  # type: ignore # Pre-trained VGG16 model
from tensorflow.keras.models import Model  # type: ignore # Keras Model class
from tensorflow.keras.layers import Dense, Dropout, Flatten  # type: ignore # Layers to add to the model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # Data augmentation
from tensorflow.keras.optimizers import Adam  # type: ignore # Optimizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore # Callbacks for training
from sklearn.metrics import confusion_matrix, classification_report  # Metrics for evaluation

# Define paths
base_directory = "/set/path/to/images/dataset"
csv_directory = "/set/path/to/csv/folder"
model_save_path = "/set/path/to/model/saving/folder/art_style_model.h5"

# Load training and validation data
train_df = pd.read_csv(os.path.join(csv_directory, 'small_style_train.csv'), header=None)
val_df = pd.read_csv(os.path.join(csv_directory, 'small_style_val.csv'), header=None)

# Set column names
train_df.columns = ['filename', 'style']
val_df.columns = ['filename', 'style']

# Convert style labels to string type
train_df['style'] = train_df['style'].astype(str)
val_df['style'] = val_df['style'].astype(str)

# Define data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define data augmentation for validation images (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=base_directory,
    x_col='filename',
    y_col='style',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=base_directory,
    x_col='filename',
    y_col='style',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the VGG16 base model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Define the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    verbose=2,
    callbacks=[early_stopping, reduce_lr]
)

# Save the trained model
model.save(model_save_path)

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Reset the validation generator and make predictions
val_generator.reset()
predictions = model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes

# Compute and plot the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
style_names = [k for k, v in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=style_names, yticklabels=style_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Styles')
plt.ylabel('True Styles')
plt.show()

# Print classification report
print('Classification Report')
print(classification_report(true_classes, predicted_classes, target_names=style_names))