# Image Classification Using Perceptron, ANN, and CNN

## Project Overview

This project focuses on image classification using three different machine learning and deep learning approaches: Perceptron, Artificial Neural Network (ANN), and Convolutional Neural Network (CNN).

The main purpose of this project was to understand how different models process image data and how their architectures affect their ability to classify images into different categories.

Through this project, I learned the complete workflow of an image classification problem, starting from data preprocessing and normalization to model building, training, evaluation, visualization, and comparison.

---

## Objectives

The main objectives of this project were:

- To understand the basic workflow of an image classification problem.
- To preprocess image data before providing it to machine learning models.
- To understand how pixel values are normalized.
- To convert image labels into a format suitable for neural networks.
- To understand how a simple Perceptron-based classifier works.
- To understand the architecture and working of an Artificial Neural Network.
- To understand why Convolutional Neural Networks are suitable for image-related tasks.
- To train and evaluate different models on the same dataset.
- To compare the performance of Perceptron, ANN, and CNN.
- To analyze model predictions using visualization techniques and a confusion matrix.

---

## Dataset

The project uses a grayscale image dataset containing images of size 28 × 28 pixels.

Each image is represented using pixel values ranging from 0 to 255. The dataset contains multiple classes, and each image has a corresponding label indicating the class to which it belongs.

Since the images are grayscale, each image contains only one channel.

Before training the models, the image data was reshaped and normalized so that it could be processed efficiently by the neural networks.

---

## Technologies and Libraries Used

The project was implemented using Python.

The main libraries used in this project are:

- **NumPy** – Used for numerical operations and manipulating image arrays.
- **Pandas** – Used for loading and working with tabular dataset information.
- **Matplotlib** – Used for creating graphs and visualizing model performance.
- **Seaborn** – Used for creating the confusion matrix heatmap.
- **Scikit-learn** – Used for preprocessing, data splitting, the Perceptron model, and evaluation metrics.
- **TensorFlow/Keras** – Used for building, training, and evaluating ANN and CNN models.

---

## Data Preprocessing

Data preprocessing is an important step before training a machine learning model.

First, the label column was separated from the input features. The image pixels were stored as input data, while the corresponding labels were stored as target data.

The pixel values originally ranged from 0 to 255. These values were converted into floating-point numbers and normalized to a range between 0 and 1.

This was done using:

`X_train = X_train.astype("float32") / 255.0`

Normalization makes the numerical values smaller and puts the input data on a consistent scale, which helps neural networks during training.

The same preprocessing was applied to both training and testing data.

---

## Image Reshaping

The original image data was stored as a one-dimensional array containing 784 pixel values.

Since 28 × 28 = 784, the data was reshaped into a 28 × 28 image representation.

For ANN-based models, the images were represented as:

`(number_of_images, 28, 28)`

For the CNN model, an additional channel dimension was added:

`(number_of_images, 28, 28, 1)`

The value `1` represents the single grayscale channel.

---

## Label Encoding

The class labels were converted into one-hot encoded vectors using `to_categorical()`.

For example, if there are 10 classes, a label such as `3` is represented as a vector where the fourth position contains `1` and the remaining positions contain `0`.

One-hot encoding allows the neural network to represent the target as a probability distribution across multiple classes.

---

# Models Implemented

## 1. Perceptron-Based Model

The first model is a simple fully connected classifier consisting of a Flatten layer and an output Dense layer.

The Flatten layer converts the 28 × 28 image into a one-dimensional vector containing 784 values.

The Dense layer contains 10 neurons because the dataset contains 10 classes.

A Softmax activation function is used in the output layer to convert the model's outputs into probabilities for each class.

The class with the highest probability is selected as the predicted class.

### Architecture

`28 × 28 Image → Flatten → Dense(10) → Softmax`

### Purpose

This model provides a simple baseline for understanding image classification.

It demonstrates how a basic fully connected model can process image pixels without explicitly learning spatial features.

---

## 2. Artificial Neural Network

The second model is an Artificial Neural Network containing multiple fully connected layers.

The input image is first flattened into a one-dimensional vector.

The network then contains:

- A Dense layer with 128 neurons.
- A Dense layer with 64 neurons.
- A final Dense layer with 10 neurons.
- ReLU activation functions are used in the hidden layers.
- Softmax is used in the output layer.

### Architecture

`28 × 28 Image → Flatten → Dense(128) → Dense(64) → Dense(10) → Softmax`

### Purpose

The ANN is more powerful than the simple Perceptron model because it contains multiple hidden layers that allow it to learn more complex patterns from the input data.

ANNs are commonly used for classification, regression, prediction, and many other machine learning tasks.

---

## 3. Convolutional Neural Network

The third model is a Convolutional Neural Network.

CNNs are particularly useful for image-related tasks because they can learn spatial patterns and local features from images.

The CNN used in this project contains convolutional layers, max-pooling layers, a flattening layer, fully connected layers, and dropout.

### Architecture

`28 × 28 × 1 Image`

→ `Conv2D(32)`

→ `MaxPooling2D`

→ `Conv2D(64)`

→ `MaxPooling2D`

→ `Flatten`

→ `Dense(128)`

→ `Dropout(0.5)`

→ `Dense(10)`

→ `Softmax`

### Convolutional Layers

The convolutional layers use learnable filters to detect useful patterns in an image.

The first convolutional layer can learn simple features such as edges and basic shapes.

The deeper convolutional layer can combine these patterns and learn more complex features.

### Max Pooling

Max pooling reduces the spatial dimensions of the feature maps.

It keeps the strongest activation from each pooling region and reduces the amount of computation required by later layers.

### Dropout

Dropout is a regularization technique used to reduce overfitting.

During training, some neurons are randomly deactivated so that the network does not become overly dependent on particular neurons.

---

# Model Training

The models were trained using the `fit()` function.

The training process was performed for multiple epochs using mini-batches.

An epoch represents one complete pass through the training dataset.

The batch size determines how many training examples are processed before the model updates its weights.

The models also recorded training and validation accuracy and loss during training.

---

# Optimizers and Loss Function

The Perceptron model was trained using Stochastic Gradient Descent (SGD).

The ANN and CNN models were trained using the Adam optimizer.

The models used categorical cross-entropy as the loss function because this is a multi-class classification problem with one-hot encoded labels.

Accuracy was used as a performance metric to measure how many predictions were correct.

---

# Model Evaluation

After training, each model was evaluated using the test dataset.

The `evaluate()` function was used to calculate the model's loss and accuracy.

The final test accuracy was then extracted and converted into a percentage for comparison.

This allowed the performance of the three models to be compared using the same test dataset.

---

# Visualization and Analysis

Several visualization techniques were used to understand the behavior of the models.

## Training and Validation Curves

Training and validation accuracy were plotted across epochs to observe how the models learned over time.

Training and validation loss were also plotted.

These graphs help identify whether the model is learning properly and can also provide indications of overfitting.

---

## Validation Accuracy Comparison

The validation accuracy of the Perceptron, ANN, and CNN models was plotted on the same graph.

This visualization makes it easier to observe how the performance of the different architectures changes across training epochs.

---

## Prediction Visualization

Random test images were selected and displayed along with their actual labels.

The predictions made by the Perceptron, ANN, and CNN models were then displayed for comparison.

This provides a direct visual understanding of how different models classify the same image.

---

## Confusion Matrix

A confusion matrix was created for the CNN model.

The rows represent the actual classes, while the columns represent the predicted classes.

The diagonal values represent correctly classified images.

The off-diagonal values represent incorrect predictions and show which classes are being confused with each other.

A heatmap was used to make the confusion matrix easier to interpret.

---

# What I Learned

Through this project, I learned the complete workflow of an image classification problem.

I learned how to:

- Load and inspect a dataset using Pandas.
- Separate input features and target labels.
- Understand the shape of image data.
- Normalize pixel values.
- Reshape image data for different types of models.
- Convert class labels into one-hot encoded vectors.
- Build a simple fully connected classification model.
- Understand the architecture of an Artificial Neural Network.
- Understand why CNNs are effective for image classification.
- Understand convolution and feature extraction at a basic level.
- Understand the purpose of pooling layers.
- Understand how Flatten converts feature maps into a vector.
- Understand how Dropout helps reduce overfitting.
- Use different optimizers such as SGD and Adam.
- Use categorical cross-entropy for multi-class classification.
- Train models using epochs and batches.
- Evaluate trained models using test data.
- Plot training and validation accuracy.
- Plot training and validation loss.
- Compare different models visually.
- Interpret predictions made by different models.
- Understand and interpret a confusion matrix.

---

# Real-World Applications

The concepts learned in this project are useful in many real-world applications.

### Image Classification

CNN-based image classification can be used to identify objects, handwritten digits, animals, products, and other categories of images.

### Medical Imaging

Deep learning models can be used to assist in analyzing medical images such as X-rays, CT scans, and MRI images.

### Face and Object Recognition

CNNs are widely used as components of systems that detect and recognize faces, objects, and visual patterns.

### Autonomous Systems

Computer vision models can help autonomous systems understand objects and environments captured through cameras.

### Document and Handwriting Recognition

Image classification techniques can be used for recognizing handwritten characters, digits, and documents.

---

# Project Workflow

The overall workflow of the project was:

`Dataset`

↓

`Data Inspection`

↓

`Feature and Label Separation`

↓

`Image Normalization`

↓

`Image Reshaping`

↓

`Label Encoding`

↓

`Model Building`

↓

`Model Training`

↓

`Model Evaluation`

↓

`Visualization`

↓

`Model Comparison`

---

# Results

The project compares three different approaches for image classification:

- Perceptron
- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)

The final test accuracies of the models were visualized using a bar chart.

The training and validation curves were also analyzed to understand the learning behavior of the models.

The confusion matrix was used to examine the classification behavior of the CNN model across different classes.

---

# Conclusion

This project helped me understand the practical workflow of building an image classification system using machine learning and deep learning techniques.

Starting with a simple fully connected classifier helped me understand the basic idea of classification. Building an ANN helped me understand how multiple layers can learn more complex patterns. Finally, implementing a CNN helped me understand why convolutional architectures are particularly useful for image data.

Overall, this project gave me practical experience with data preprocessing, neural network architecture, model training, evaluation, visualization, and performance analysis.

It also helped me understand how concepts such as normalization, activation functions, convolution, pooling, dropout, optimizers, loss functions, and confusion matrices are used together in a real machine learning workflow.

---

# How to Run the Project

## 1. Clone the Repository

Clone this repository to your local machine.

## 2. Install Required Libraries

Install the required Python libraries using:

`pip install numpy pandas matplotlib seaborn scikit-learn tensorflow`

## 3. Open the Notebook

Open the Jupyter Notebook containing the project code.

## 4. Run the Notebook

Run the cells sequentially to perform data preprocessing, train the models, evaluate their performance, and generate the visualizations.
