import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.metrics import confusion_matrix, classification_report

# Define paths
base_directory = "/set/path/to/images/dataset"
csv_directory = "/set/path/to/csv/folder"
model_save_path = "/set/path/to/model/saving/folder/art_style_model.h5"

# Load data and specify no header to define columns manually
train_df = pd.read_csv(os.path.join(csv_directory, 'small_style_train.csv'), header=None)
val_df = pd.read_csv(os.path.join(csv_directory, 'small_style_val.csv'), header=None)

# Assign column names
train_df.columns = ['filename', 'style']
val_df.columns = ['filename', 'style']

# Convert 'style' column to string explicitly
train_df['style'] = train_df['style'].astype(str)
val_df['style'] = val_df['style'].astype(str)

# Data augmentation configuration
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

val_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
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

# Load VGG16 as the base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Adding custom layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Final model setup
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model callbacks
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

# Save the model
model.save(model_save_path)

# Plot training results
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

# Predict the validation dataset
val_generator.reset()  # Resetting the generator is crucial before making predictions
predictions = model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes

# Generate the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
style_names = [k for k, v in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]

# Plot the confusion matrix with style names
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=style_names, yticklabels=style_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Styles')
plt.ylabel('True Styles')
plt.show()

# Print classification report
print('Classification Report')
print(classification_report(true_classes, predicted_classes, target_names=style_names))