import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25  # Slightly more epochs for the larger dataset

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Using relative paths as established in the portability task
DATASET_PATH = os.path.join(BASE_DIR, "..", "backend", "knee_dataset")
NON_XRAY_PATH = os.path.join(DATASET_PATH, "non_xray")
XRAY_TRAIN_PATH = os.path.join(DATASET_PATH, "train")
XRAY_VAL_PATH = os.path.join(DATASET_PATH, "val")

def build_binary_model():
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 1)), # Grayscale
        
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid') # Binary
    ])
    return model

def get_file_lists():
    # 1. Non-Xray files
    non_xray_files = [os.path.join(NON_XRAY_PATH, f) for f in os.listdir(NON_XRAY_PATH) if f.endswith('.jpg')]
    random.shuffle(non_xray_files)
    
    # Split non-xrays 80/20 for train/val
    split_idx = int(0.8 * len(non_xray_files))
    non_xray_train = non_xray_files[:split_idx]
    non_xray_val = non_xray_files[split_idx:]
    
    # 2. X-ray training files (all grades)
    xray_train_files = []
    for grade in os.listdir(XRAY_TRAIN_PATH):
        gp = os.path.join(XRAY_TRAIN_PATH, grade)
        if os.path.isdir(gp):
            xray_train_files.extend([os.path.join(gp, f) for f in os.listdir(gp) if f.endswith(('.png', '.jpg'))])
            
    # 3. X-ray validation files
    xray_val_files = []
    for grade in os.listdir(XRAY_VAL_PATH):
        gp = os.path.join(XRAY_VAL_PATH, grade)
        if os.path.isdir(gp):
            xray_val_files.extend([os.path.join(gp, f) for f in os.listdir(gp) if f.endswith(('.png', '.jpg'))])
            
    print(f"X-ray Train: {len(xray_train_files)} | Val: {len(xray_val_files)}")
    print(f"Non-Xray Train: {len(non_xray_train)} | Val: {len(non_xray_val)}")
    
    return (xray_train_files, non_xray_train), (xray_val_files, non_xray_val)

def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

def create_dataset(x_files, n_files):
    all_files = x_files + n_files
    all_labels = [1] * len(x_files) + [0] * len(n_files)
    
    # Shuffle
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)
    
    ds = tf.data.Dataset.from_tensor_slices((list(all_files), list(all_labels)))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=min(len(all_files), 2000)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    train_lists, val_lists = get_file_lists()
    
    train_ds = create_dataset(*train_lists)
    val_ds = create_dataset(*val_lists)
    
    # Calculate Class Weights to handle imbalance (5778 vs ~1600)
    # Weight = Total / (NumClasses * NumSamplesInClass)
    total_samples = len(train_lists[0]) + len(train_lists[1])
    weight_for_xray = total_samples / (2 * len(train_lists[0]))
    weight_for_non = total_samples / (2 * len(train_lists[1]))
    
    class_weight = {0: weight_for_non, 1: weight_for_xray}
    print(f"Class Weights: {class_weight}")
    
    model = build_binary_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = os.path.join(BASE_DIR, "..", "ml_model", "binary_xray_validator.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    ]
    
    print("Starting Binary Training with Full X-ray Dataset...")
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        class_weight=class_weight
    )
    
    model.save(checkpoint_path)
    print(f"Retraining complete. Model saved to {checkpoint_path}")
