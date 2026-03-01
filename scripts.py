import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import class_weight

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ======================================================
# DATA PATH
# ======================================================
data_path = os.path.dirname(os.path.abspath(__file__))
train_images_path = os.path.join(data_path, "train_images")

# ======================================================
# PARAMETERS
# ======================================================
IMG_SIZE = 260  
BATCH_SIZE = 32
EPOCHS_WARMUP = 12
EPOCHS_FINE_TUNE = 25

# ======================================================
# DATA LOADING
# ======================================================
print("Loading datasets with optimized pipeline...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_images_path,
    validation_split=0.2, # 20% validation for better verification
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_images_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")

# --- CALCULATE CLASS WEIGHTS (PREVENT UNDERFITTING RARE CLASSES) ---
print("Calculating class weights...")
all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(np.argmax(labels.numpy()))

class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
class_weights_dict = dict(enumerate(class_weights_vals))
print(f"Class Weights applied: {class_weights_dict}")

# Optimize performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ======================================================
# ROBUST DATA AUGMENTATION
# ======================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="data_augmentation")

# ======================================================
# BUILD "PREMIUM" MODEL Architecture
# ======================================================
def build_premium_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Augmentation
    x = data_augmentation(inputs)
    
    # Base EfficientNetV2 (Pre-scaling internal)
    base_model = EfficientNetV2B2(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    
    # Freeze by default
    base_model.trainable = False
    
    # Premium Head (Robust against over/underfitting)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Optimized Dense Layer
    x = layers.Dense(256, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

model, base_model = build_premium_model(num_classes)

# Print Keras Summary Table
model.summary()

# ======================================================
# PHASE 1: WARMUP (3 Epochs)
# ======================================================
print("\nStarting Phase 1: Warmup (3 Epochs)...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_paddy_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ======================================================
# PHASE 2: FINE-TUNING (7 Epochs)
# ======================================================
print("\nStarting Phase 2: Fine-Tuning (7 Epochs)...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), # Slightly higher than before to reach 90% in 10 total epochs
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=7,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ======================================================
# SAVE & METADATA
# ======================================================
model.save("final_paddy_model.keras")
labels = dict(enumerate(class_names))

with open('paddy_model_metadata.pkl', 'wb') as f:
    pickle.dump({
        'model_path': "final_paddy_model.keras",
        'labels': labels,
        'img_size': IMG_SIZE
    }, f)

print("\nTraining Completed! ✅")
print("Best model saved as final_paddy_model.keras")

# ======================================================
# PLOT PROGRESS
# ======================================================
def plot_results(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    
    plt.figure(figsize=(8, 4))
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.axvline(x=len(h1.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning start')
    plt.title('Premium Model Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

try:
    plot_results(history1, history2)
except Exception as e:
    print(f"Plotting skipped: {e}")