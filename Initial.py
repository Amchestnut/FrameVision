import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

# For training
from tensorflow import keras
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping


# Step 1: I will split the dataset
def split_the_dataset(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    categories = os.listdir(dataset_dir)

    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            train, validation = train_test_split(images, test_size=0.1, random_state=42)
            os.makedirs(os.path.join(output_dir, 'train', category), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'validation', category), exist_ok=True)
            # I use this shutil to copy images into the appropriate folders
            for image in train:
                shutil.copy(os.path.join(category_path, image), os.path.join(output_dir, 'train', category))
            for image in validation:
                shutil.copy(os.path.join(category_path, image), os.path.join(output_dir, 'validation', category))
            print('Dataset split successfully')

# I can print to check if everything was split 90/10 as we wanted
def print_the_split(output_dir):
    for split in ['train', 'validation']:
        print(f"{split.upper()} DATASET:")
        split_path = os.path.join(output_dir, split)
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            print(f"  {category}: {len(os.listdir(category_path))} images")



# 2) Data augmentation:
# I literally NEVER use the original images for training, i augment them, make distorted (a bit) copies, and train the model ON THEM.
# This is very cool
# we preprocess the image DATA and AUGMENT the training images to make the model more robust to variations

def prepare_data(train_directory, validation_directory):
    # Data augmentation for TRAINING.
    # ImageDataGenerator is a class in KERAS that preprocess image data and applies transformations to augment it
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,    # Normalize pixel values.   Pixel values from [0, 255] (original image range) to [0, 1] to make training numerically stable
        # rotation_range=20,      # Randomly rotate images
        # zoom_range=0.2,         # Randomly zoom in/out
        # horizontal_flip=True    # Randomly flip images horizontally
        rotation_range=35,
        zoom_range=0.35,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=True
    )

    # Rescaling for VALIDATION
    # I dont augment the validation set because it is used to evaluate the model's performance, and augmentation might distort this evaluation.
    # We just do normalisation of pixels to [0, 1] just like in training data
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Data generators
    # 'flow_from_directory': Automatically label images based on their folder names
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(224, 224),
        batch_size=32,              # Process 32 images at a time, so we balance memory usage and training speed
        class_mode='categorical'    # categorical because Each image is labeled as one of 45 categories
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator



# 3) build the Transfer Learning model
# for this I will use "ResNet50" as the base model, freeze its layers, and add a custom classification head
def build_model():
    # First: load the ResNet50 model (pre-trained on ImageNet)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # base_model.trainable = False    # Freeze the base model layers

    # Freezing only the first 20 layers ensures lower-level features remain stable, avoiding overfitting and keeping training efficient.
    base_model.trainable = True
    for layer in base_model.layers[:20]:
        layer.trainable = False

    # Here I add a custom classification head
    model = Sequential([
        base_model,
        Flatten(),      # Flatten the feature maps
        Dense(512, activation='relu'),
        Dropout(0.6),   # The Dropout(0.5) layers reduce overfitting by randomly dropping 50% of neurons during training.
        Dense(256, activation='relu'),
        Dropout(0.6),   # This prevents overfitting
        Dense(45, activation='softmax')    # '45' as the number of classes for our RESISC45
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


# 4) TRAINING
def train_model(model, train_generator, validation_generator, epochs):
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping]
        # steps_per_epoch=train_generator.samples // train_generator.batch_size,
        # validation_steps=validation_generator.samples // validation_generator.batch_size
    )
    return history



# 5) EVALUATION
def plot_history(history):
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



dataset_dir = 'NWPU-RESISC45'
output_dir = 'dataset/'
# split_the_dataset(dataset_dir, output_dir)    Already did this
# print_the_split(output_dir)


# Paths
train_directory = 'dataset/train'
val_directory = 'dataset/validation'

# print("GPU available:", tf.config.list_physical_devices('GPU'))

# Step 1: Prepare data
train_generator, val_generator = prepare_data(train_directory, val_directory)

# Step 2: Build model
model = build_model()

# Step 3: Train model
history = train_model(model, train_generator, val_generator, epochs=100)

# Step 4: Evaluate performance
plot_history(history)

# Step 5: I want to save this model
os.makedirs('models', exist_ok=True)
model.save('models/resisc45_model_v3.keras')


