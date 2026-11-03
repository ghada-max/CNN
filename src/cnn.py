import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1 Load the CIFAR-10 Dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# 2 Visualize the First Image (plt.figure must go BEFORE plt.imshow)
plt.figure(figsize=(3, 3))
plt.imshow(X_train[0])
plt.axis('off')
plt.show()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
X_train = X_train / 255.0
X_test = X_test / 255.0


# 3 ARTIFICIAL NEURAL NETWORK 

print("--- Training ANN Model ---")
ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the ANN
ann.fit(X_train, y_train, epochs=5)


# 5. CONVOLUTIONAL NEURAL NETWORK (CNN)
print("\n--- Training CNN Model ---")
cnn = models.Sequential([
    # First Convolution & Pooling Block
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolution & Pooling Block
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening and Dense Classification Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the CNN
cnn.fit(X_train, y_train, epochs=5)