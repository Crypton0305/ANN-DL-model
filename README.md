# MNIST Handwritten Digit Classifier

A simple Deep Learning project that trains an Artificial Neural Network (ANN)
on the MNIST dataset and lets users test it through a Streamlit web app.

Upload a picture of a handwritten digit, and the app will:
- Show the uploaded image
- Predict the digit (0-9)
- Show the model's confidence
- Let you enter the actual digit to check if the prediction was correct

---

## Project Files

| File               | Purpose                                                   |
|--------------------|------------------------------------------------------------|
| `train_model.py`   | Builds and trains the ANN model, saves it as `mnist_model.keras` |
| `app.py`           | Streamlit web app for uploading images and viewing predictions |
| `requirements.txt` | Python packages needed to run the project                 |
| `runtime.txt`      | Pins the Python version (3.11) for deployment platforms    |

---

## Model Architecture

- Input: 784 values (28x28 flattened image)
- Hidden Layer 1: 256 neurons, ReLU activation
- Dropout: 20%
- Hidden Layer 2: 128 neurons, ReLU activation
- Dropout: 20%
- Output: 10 neurons (digits 0-9), Softmax activation

Trained for 15 epochs, using the Adam optimizer and sparse categorical
crossentropy loss.

---

## How to Run Locally

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (this creates `mnist_model.keras`):
   ```bash
   python train_model.py
   ```

3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal (usually `http://localhost:8501`)

> Note: If `mnist_model.keras` is missing when the app starts, it will
> automatically train a new model in the background before showing the UI.

---

## Deploying on Streamlit Community Cloud

1. Push all project files to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app.
3. Point it to your repository and set the main file as `app.py`.
4. The `runtime.txt` file ensures Python 3.11 is used, since TensorFlow does
   not yet support newer Python versions used by default on some platforms.

---

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow (PIL)

---

## How Predictions Work

1. The uploaded image is converted to grayscale.
2. It is resized to 28x28 pixels (same size as MNIST training images).
3. If the background is light-colored, the image is inverted so digits
   appear as white strokes on a black background (matching MNIST style).
4. Pixel values are normalized between 0 and 1.
5. The image is flattened and passed to the model for prediction.
