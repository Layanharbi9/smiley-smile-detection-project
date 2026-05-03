import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

TRAIN_DIR = "train_folder"  
TEST_DIR  = "test_folder"   
IMG_SIZE    = 224       
BATCH_SIZE  = 32        
EPOCHS      = 20      
LEARNING_RATE = 0.0001  
MODEL_SAVE_PATH = "smile_model.h5"   

print("=" * 60)
print("  Smiley - Smile Detection Training")
print("=" * 60)
print("\n[1] Loading and augmenting dataset...")


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,        
    rotation_range=15,           
    width_shift_range=0.1,        
    height_shift_range=0.1,       
    horizontal_flip=True,        
    zoom_range=0.1,              
    brightness_range=[0.8, 1.2],   
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',   
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\n   Classes found: {train_generator.class_indices}")
print(f"   Training images  : {train_generator.samples}")
print(f"   Testing  images  : {test_generator.samples}")

print("\n[2] Building MobileNetV2 model with Transfer Learning...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'     
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)                  
output = Dense(1, activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

print("\n[3] Training Phase 1: Feature Extraction (base model frozen)...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n[4] Training Phase 2: Fine-Tuning (unfreeze last 30 layers)...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

print(f"\n[5] Model saved to: {MODEL_SAVE_PATH}")

def plot_history(h1, h2=None):
    acc     = h1.history['accuracy']
    val_acc = h1.history['val_accuracy']
    loss    = h1.history['loss']
    val_loss= h1.history['val_loss']

    if h2:
        acc     += h2.history['accuracy']
        val_acc += h2.history['val_accuracy']
        loss    += h2.history['loss']
        val_loss+= h2.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy', color='steelblue')
    plt.plot(epochs_range, val_acc, label='Val Accuracy', color='coral')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss', color='steelblue')
    plt.plot(epochs_range, val_loss, label='Val Loss', color='coral')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("Training curves saved to: training_curves.png")

plot_history(history, history_ft)

print("\n Training complete!")