"""
MNIST Digit Classifier - Streamlit App
------------------------------------------
This app lets the user:
1. Upload a picture of a handwritten digit.
2. See the uploaded picture and the model's predicted digit.
3. (Optional) Type in the actual/real digit to check if the
   model predicted correctly or not.

Before running this file, make sure you have already run train_model.py
so that "mnist_model.keras" exists in the same folder.

Run this app with:
    streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from PIL import Image

MODEL_PATH = "mnist_model.keras"


# Step 1: Train a new model from scratch
def train_new_model():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 784)).astype("float32") / 255
    test_images = test_images.reshape((10000, 784)).astype("float32") / 255

    new_model = keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax")
    ])

    new_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    new_model.fit(
        train_images,
        train_labels,
        epochs=15,
        batch_size=128,
        validation_split=0.1
    )

    new_model.save(MODEL_PATH)
    return new_model


# Step 2: Load the model, training it first if it does not exist yet
# @st.cache_resource makes sure this runs only once, not on every click
@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    else:
        with st.spinner("No saved model found. Training a new one, this may take a few minutes..."):
            return train_new_model()


model = load_trained_model()


# Step 2: Helper function to prepare the uploaded image for the model
def preprocess_image(pil_image):
    # Convert to grayscale (model expects 1 color channel, not RGB)
    gray_image = pil_image.convert("L")

    # Resize to 28x28, same size as MNIST training images
    resized_image = gray_image.resize((28, 28))

    image_array = np.array(resized_image).astype("float32")

    # MNIST digits are white strokes on a black background.
    # If the uploaded photo has a light/white background, colors are
    # flipped compared to MNIST, so we invert them here.
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Normalize pixel values to 0-1, same as training data
    image_array = image_array / 255.0

    # Flatten to shape (1, 784) because the model expects a flat vector
    image_array = image_array.reshape(1, 784)

    return image_array


# Step 3: Page setup
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("MNIST Handwritten Digit Classifier")
st.write("Upload a picture of a handwritten digit (0-9) to see the model's prediction.")

# Step 4: Upload section
uploaded_file = st.file_uploader("Upload Digit Image", type=["png", "jpg", "jpeg"])

# Optional textbox for the actual digit
actual_digit_input = st.text_input("Actual Digit (optional)", placeholder="e.g. 5")

predict_button = st.button("Predict", type="primary")

# Step 5: Run prediction when button is clicked
if predict_button:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        pil_image = Image.open(uploaded_file)

        # Show the uploaded image
        st.image(pil_image, caption="Uploaded Image", width=200)

        # Preprocess and predict
        processed_image = preprocess_image(pil_image)
        prediction = model.predict(processed_image)

        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        st.subheader(f"Predicted Digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Step 6: Check correctness only if user provided the actual digit
        if actual_digit_input.strip() != "":
            try:
                actual_digit_number = int(actual_digit_input)
                if actual_digit_number == predicted_digit:
                    st.success(f"Correct! Model predicted {predicted_digit} correctly.")
                else:
                    st.error(f"Wrong. You entered {actual_digit_number}, model predicted {predicted_digit}.")
            except ValueError:
                st.warning("Please enter a valid digit (0-9).")
        else:
            st.info("Enter the actual digit above to check correctness.")