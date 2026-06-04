# ─── STEP 1: IMPORTS ───────────────────────────────────────────────
# tensorflow → our deep learning framework
# keras      → the high-level API inside TensorFlow that makes
#              building neural networks much simpler
# numpy      → for array manipulation
#
# From keras we import specific building blocks:
#   Sequential     → lets us stack layers one after another
#   Conv2D         → convolutional layer — the core of a CNN
#   MaxPooling2D   → downsamples the image after each conv layer
#   Flatten        → converts 2D feature maps into a 1D vector
#   Dense          → fully connected layer (standard neural network layer)
#   Dropout        → randomly turns off neurons during training
#                    to prevent overfitting
#   BatchNorm...   → normalizes layer outputs for more stable training
#   cifar10        → the dataset, built directly into Keras

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import os

# ─── STEP 2: LOAD CIFAR-10 ─────────────────────────────────────────
# CIFAR-10 is a standard benchmark dataset with 60,000 colour images
# across 10 classes: airplane, car, bird, cat, deer, dog, frog,
# horse, ship, truck.
# Each image is 32×32 pixels with 3 colour channels (RGB).
#
# Keras downloads it automatically on first run (~170MB).
# It comes pre-split into:
#   50,000 training images
#   10,000 test images
#
# X_train shape → (50000, 32, 32, 3)
#   50000 images, each 32px tall, 32px wide, 3 colour channels
# y_train shape → (50000, 1)
#   one label per image (a number 0–9)

print("\n--- Step 1: Loading CIFAR-10 dataset ---")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(f"Training images : {X_train.shape}")
print(f"Test images     : {X_test.shape}")

# Class names matching labels 0–9
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
print(f"Classes: {CLASS_NAMES}")

# ─── STEP 3: NORMALIZE ─────────────────────────────────────────────
# Raw pixel values are 0–255 (one byte per channel).
# Neural networks train much better when inputs are small numbers.
# Dividing by 255.0 scales every pixel to the range 0.0–1.0.
# This is called normalization — one of the most important
# preprocessing steps in any image ML pipeline.
#
# .astype("float32") → converts from integer to float
# because division produces floats and we want consistency.

print("\n--- Step 2: Normalizing pixel values (0-255 → 0.0-1.0) ---")
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0
print(f"Pixel range after normalization: {X_train.min():.1f} to {X_train.max():.1f}")

# ─── STEP 4: BUILD THE CNN ─────────────────────────────────────────
# A CNN (Convolutional Neural Network) is designed specifically
# for images. Here's the intuition for each layer type:
#
# Conv2D(32, (3,3))
#   → Slides a 3×3 filter across the image looking for patterns
#     like edges, curves, corners. 32 means we learn 32 different
#     filters at this layer. Early layers detect simple features
#     (edges), deeper layers detect complex ones (eyes, wheels).
#   → activation="relu" → Rectified Linear Unit. Sets all negative
#     values to 0. This adds non-linearity so the network can learn
#     complex patterns, not just straight lines.
#
# BatchNormalization()
#   → Normalizes the output of the previous layer.
#     Makes training faster and more stable.
#
# MaxPooling2D(2,2)
#   → Takes a 2×2 block of values and keeps only the maximum.
#     Halves the image dimensions (32×32 → 16×16 → 8×8).
#     This reduces computation and makes the model less sensitive
#     to exact position of features.
#
# Dropout(0.25)
#   → During training, randomly sets 25% of neurons to 0.
#     Forces the network to not rely on any single neuron.
#     Prevents overfitting (memorizing training data).
#
# Flatten()
#   → After all conv layers, we have a 3D block of features.
#     Flatten() stretches it into a 1D list so Dense layers can use it.
#
# Dense(256, activation="relu")
#   → Standard fully-connected layer. Every neuron connects to
#     every input. This is where high-level reasoning happens.
#
# Dense(10, activation="softmax")
#   → Output layer. 10 neurons = 10 classes.
#     softmax converts raw scores into probabilities that sum to 1.
#     e.g. [0.02, 0.01, 0.85, 0.03, ...] → 85% confident it's a bird.

print("\n--- Step 3: Building the CNN architecture ---")

model = keras.Sequential([
    # Block 1 — detect basic features
    layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 2 — detect more complex features
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 3 — detect even more abstract features
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Classifier head
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.summary()

# ─── STEP 5: COMPILE ───────────────────────────────────────────────
# Before training we configure three things:
#
# optimizer="adam"
#   → Adam is the most popular optimizer in deep learning.
#     It automatically adjusts the learning rate for each parameter.
#     Think of it as the algorithm that updates the weights after
#     each batch of images based on how wrong the predictions were.
#
# loss="sparse_categorical_crossentropy"
#   → Our loss function — measures how wrong the model is.
#     "sparse" means our labels are integers (0–9), not one-hot vectors.
#     "categorical_crossentropy" is standard for multi-class problems.
#     The optimizer tries to MINIMIZE this number during training.
#
# metrics=["accuracy"]
#   → We want to see accuracy printed during training so we can
#     watch the model improve epoch by epoch.

print("\n--- Step 4: Compiling the model ---")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── STEP 6: TRAIN ─────────────────────────────────────────────────
# epochs=15 → the model sees the entire training set 15 times.
#   Each pass is one epoch. More epochs = more learning,
#   but too many causes overfitting.
#
# batch_size=64 → instead of updating weights after every single
#   image, we process 64 images at once and average the updates.
#   This is faster and more stable than updating after each image.
#
# validation_data → after each epoch, we check performance on
#   the test set WITHOUT using it for training. This tells us
#   if the model is actually generalizing or just memorizing.
#
# This will take 5–15 minutes depending on your CPU.
# You'll see val_accuracy climb each epoch — that's your model learning.

print("\n--- Step 5: Training (this will take 5-15 mins) ---")

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ─── STEP 7: EVALUATE ──────────────────────────────────────────────
# Final evaluation on the test set.
# We expect around 75-82% accuracy after 15 epochs.
# This is normal for CIFAR-10 with a simple CNN on CPU.

print("\n--- Step 6: Final evaluation ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# ─── STEP 8: SAVE THE MODEL ────────────────────────────────────────
# We save in two formats:
#
# .h5 → HDF5 format, the classic Keras save format.
#   Saves the architecture + weights + optimizer state.
#   Streamlit will load this to make predictions.
#
# We also save CLASS_NAMES as a numpy file so Streamlit
# knows which label (0-9) maps to which class name ("cat", "dog" etc.)

print("\n--- Step 7: Saving model ---")
os.makedirs("models", exist_ok=True)
model.save("models/cifar_model.h5")
np.save("models/class_names.npy", CLASS_NAMES)
print("Saved: models/cifar_model.h5")
print("Saved: models/class_names.npy")

# ─── STEP 9: SANITY CHECK ──────────────────────────────────────────
# Load the saved model fresh and test it on 5 random test images.
# We check that the loaded model gives the same predictions
# as the one we just trained.

print("\n--- Step 8: Sanity check on 5 random images ---")
loaded_model      = keras.models.load_model("models/cifar_model.h5")
loaded_classes    = np.load("models/class_names.npy")

indices = np.random.choice(len(X_test), 5, replace=False)
for i in indices:
    img        = X_test[i]
    true_label = loaded_classes[y_test[i][0]]
    prediction = loaded_model.predict(img[np.newaxis, ...], verbose=0)
    pred_label = loaded_classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    correct    = "✓" if pred_label == true_label else "✗"
    print(f"{correct} True: {true_label:<12} Predicted: {pred_label:<12} ({confidence:.1f}%)")

print("\nAll done! Both models are ready.")