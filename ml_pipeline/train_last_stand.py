
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import tf_apply_clahe, get_augmentation_model
import numpy as np

# -----------------------------
# Configuration (EfficientNet Last Stand @ 224x224)
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 8             # Reduced for max stability and memory headroom
EPOCHS = 100
NUM_CLASSES = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "backend", "knee_dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "ml_model", "knee_oa_efficientnet_last_stand.h5")

def prepare_dataset(subset):
    path = os.path.join(DATA_DIR, subset)
    # Load unbatched for clean mapping
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=None,
        shuffle=(subset == 'train')
    )
    
    # Ordinal Encoding
    def to_ordinal(image, label):
        grade = tf.argmax(label, axis=-1)
        mask = tf.range(1, 5, dtype=tf.int64)
        ordinal_label = tf.cast(grade >= mask, tf.float32)
        # Re-assert shapes for TF graph
        image.set_shape((*IMG_SIZE, 3))
        ordinal_label.set_shape((4,))
        return image, ordinal_label

    ds = ds.map(to_ordinal, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(tf_apply_clahe, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and Prefetch
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_last_stand_model():
    base_model = tf.keras.applications.EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    # EfficientNetV2 training is more stable if we start with a slow learning rate
    base_model.trainable = True 
    
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    
    # Augmentation
    aug = get_augmentation_model()
    x = aug(inputs)
    
    # Backbone (EfficientNetV2 expects [0, 255])
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Final Layer with L2
    outputs = layers.Dense(
        4, 
        activation='sigmoid', 
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    
    model = models.Model(inputs, outputs)
    
    # Cosine Decay with Warmup logic (simplified)
    # 2e-5 is a safe starting point for EffNetV2S fine-tuning
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-5, 
        decay_steps=EPOCHS * (5778 // BATCH_SIZE),
        alpha=0.1
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['binary_accuracy']
    )
    return model

if __name__ == "__main__":
    print(f"🥇 Launching EfficientNet LAST STAND @ {IMG_SIZE} Native...")
    
    train_ds = prepare_dataset('train')
    val_ds = prepare_dataset('val')
    
    model = build_last_stand_model()
    # model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
    ]

    print("Starting fit...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print(f"🏁 LAST STAND COMPLETE. Model saved to: {MODEL_SAVE_PATH}")
