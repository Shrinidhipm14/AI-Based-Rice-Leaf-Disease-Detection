import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Path to your folder
data_path = 'dataset' 

# 2. Prep the 3,829+ images
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, 
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training Data (Categorical mode is the fix!)
train_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical', 
    subset='training'
)

# Validation Data
val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 3. Build the Brain (6 Diseases/Categories)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(6, activation='softmax') # Correctly matches your 6 folders
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Start Training
print(f"Nidhima, training started on {train_data.samples} images!")
print(f"Categories found: {train_data.class_indices}")

model.fit(train_data, validation_data=val_data, epochs=10)

# 5. Save the Master Brain
model.save('rice_disease_model.keras')
print("--- Master Brain Saved Successfully! ---")