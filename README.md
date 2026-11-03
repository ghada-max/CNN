# CIFAR-10 Image Classification: ANN vs. CNN from First Principles

An educational and experimental repository exploring image classification on the **CIFAR-10 dataset**, transitioning from traditional Artificial Neural Networks (MLPs) to Convolutional Neural Networks (CNNs) built from first principles.

---

## 🚀 Project Overview
The goal of this project is to understand the mathematical and structural limitations of Fully Connected layers (ANNs) when processing image data, and how Convolutional layers introduce spatial awareness, weight sharing, and parameter efficiency.

- **Dataset:** CIFAR-10 (50,000 training images, 10 distinct classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Environments:** Developed and executed using **Jupyter Notebooks**, running inside isolated **Docker** containers on Ubuntu.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Data Visualization & Processing:** Matplotlib, NumPy
- **Environment:** Docker (`tensorflow/tensorflow:latest-jupyter`) / Virtual Environments (`venv`)

---

## 🏗️ Model Architectures

### 1. Artificial Neural Network (ANN / MLP)
* Uses `layers.Flatten()` to transform the $32 \times 32 \times 3$ image grid into a 1D vector (3,072 inputs).
* Dense hidden layers with **ReLU** activation (3,000 neurons $\rightarrow$ 1,000 neurons).
* Output layer with **10 neurons** and **Softmax** activation for multi-class probability distribution.
* *Limitation:* Brute-force connections leading to millions of parameters and poor spatial translation invariance.

### 2. Convolutional Neural Network (CNN)
* Preserves spatial 2D grid structure using `Conv2D` and `MaxPooling2D` layers.
* Utilizes filter kernels and weight sharing to extract local features (edges, textures, shapes) efficiently with a fraction of the parameters.

---

