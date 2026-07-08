"""
MNIST Digit Classifier - Training Script
------------------------------------------
This script trains a simple ANN (Artificial Neural Network) on the MNIST
handwritten digit dataset. Compared to the original notebook, this version:

1. Adds an extra hidden layer + Dropout, so the model learns better
   features and generalizes well (less overfitting).
2. Reaches high accuracy in far fewer epochs (15 instead of 50).
3. Saves the trained model to disk so the Gradio app (app.py) can load
   it directly without retraining every time.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Step 2: Reshape and normalize the images
# Each image is 28x28 pixels -> flatten it to a single row of 784 numbers
train_images = train_images.reshape((60000, 784)).astype("float32") / 255
test_images = test_images.reshape((10000, 784)).astype("float32") / 255

# Step 3: Build the model
# Two hidden layers + Dropout help the model learn faster and avoid
# memorizing the training data (which was the issue with the old 50-epoch model)
model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

# Step 4: Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 5: Train the model
# Only 15 epochs are needed now to get strong accuracy without overfitting
model.fit(
    train_images,
    train_labels,
    epochs=15,
    batch_size=128,
    validation_split=0.1
)

# Step 6: Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")

# Step 7: Save the trained model to disk
# The Gradio app will load this file to make predictions
model.save("mnist_model.keras")
print("\nModel saved as mnist_model.keras")
