OCR Using CNN

This project implements Optical Character Recognition (OCR) using two approaches: a Convolutional Neural Network (CNN) and a k-Nearest Neighbors (k-NN) algorithm. The CNN is trained on the MNIST dataset for higher accuracy, while the k-NN serves as a simpler, baseline classifier for digit recognition. The system processes an input image, segments the characters, and predicts the text using the chosen algorithm.

Features

Preprocessing: Grayscale conversion, adaptive thresholding, and denoising.

Segmentation: Automatic segmentation of individual characters using contours.

k-NN Classifier: A simple distance-based classifier for digit recognition.

CNN Model: A more robust model trained on the MNIST dataset for digit classification.

End-to-End OCR Pipeline: Converts an input image into recognized text.

Prerequisites

To run this project, ensure the following dependencies are installed:

Python 3.6+

OpenCV

TensorFlow/Keras

NumPy

Matplotlib

Install the required Python libraries using:

pip install opencv-python-headless tensorflow numpy matplotlib

Dataset

The project uses the MNIST dataset, a standard dataset for handwritten digit recognition. The dataset is automatically downloaded when the code is run.

How It Works

Preprocessing:

The input image is converted to grayscale.

Adaptive thresholding is applied to binarize the image.

Median blur is applied to remove noise.

Segmentation:

Characters are segmented using contours.

Each character is resized to 28x28 pixels to match the input size of the CNN model.

k-NN Algorithm:

The k-NN algorithm is a simple, distance-based classifier.

Each segmented character is flattened into a feature vector and compared to the training data using Euclidean distance.

The class of the majority of the k-nearest neighbors is assigned to the character.

CNN Model:

A Convolutional Neural Network (CNN) is trained on the MNIST dataset.

The CNN predicts the digit for each segmented character with higher accuracy.

OCR Pipeline:

The input image is processed, characters are segmented, and either k-NN or CNN is used to predict the text.

Usage

Clone this repository:

git clone https://github.com/your-username/ocr-with-cnn.git
cd ocr-with-cnn

Install dependencies:

pip install -r requirements.txt

Run the program:

python main.py

Place your custom test image in the root directory (e.g., test.png) and modify image_path in the code to your image file name.

Testing with Example Image

To test with an MNIST example image:

Generate the example image using the provided code:

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()
example_image = x_test[0]
cv2.imwrite('mnist_example.png', example_image)
plt.imshow(example_image, cmap='gray')
plt.title('Example MNIST Image')
plt.show()

Use mnist_example.png as the input image for the OCR pipeline.

Model Architecture

The CNN model consists of:

Two convolutional layers with ReLU activation and max-pooling.

A flattening layer to convert 2D features to 1D.

Two fully connected layers, with the final layer using softmax activation for digit classification.

Results

Accuracy

k-NN Classifier:

Training Accuracy: ~90%

Validation Accuracy: ~88%

CNN Model:

Training Accuracy: ~99.4%

Validation Accuracy: ~98.9%

Example Prediction

Input Image: mnist_example.pngPredicted Text (CNN): 2

Issues and Improvements

Common Issues

k-NN struggles with curvy digits (e.g., 2 and 3) and noisy images.

Incorrect predictions for poorly segmented characters.

Difficulty recognizing characters not resembling MNIST digits.

Suggested Improvements

Train the CNN on augmented data (e.g., rotated, noisy digits).

Fine-tune the model on custom datasets for better generalization.

Add a post-processing step to handle multi-character text.

Contributing

Feel free to fork this repository and submit pull requests for enhancements or bug fixes. Contributions are welcome!

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements

MNIST Dataset

TensorFlow Documentation

OpenCV Documentation

Contact

For questions or feedback, please contact: [your-email@example.com]

