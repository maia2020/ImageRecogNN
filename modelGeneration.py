import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (
    testing_images,
    testing_labels,
) = datasets.cifar10.load_data()
# Training images is a array of pixels (images)

# Normalizing the data
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
# Object of the class that will be able to identify the images

# Visualizing the data
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()

# As this is just a simple project, there is no need to use all the data
# So we will use only 20000 images for training and 4000 for testing
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Creating the model
model = models.Sequential()
model.add(
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3))
)  # convolutional layer because we are dealing with images
model.add(
    layers.MaxPooling2D((2, 2))
)  # max pooling layer to reduce the size of the image and make it easier to process
model.add(
    layers.Conv2D(64, (3, 3), activation="relu")
)  # this time we are using 64 filters because the image is smaller
model.add(layers.MaxPooling2D((2, 2)))
model.add(
    layers.Conv2D(64, (3, 3), activation="relu")
)  # this time we are using 64 filters because the image is smaller
model.add(
    layers.Flatten()
)  # flattening the image to make it easier to process (making it a 1D array)
model.add(
    layers.Dense(64, activation="relu")
)  # dense layer to make the model more accurate
model.add(
    layers.Dense(10, activation="softmax")
)  # softmax layer so that we have a distribution of probabilities

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)  # compiling the model

model.fit(
    training_images,
    training_labels,
    epochs=10,
    validation_data=(testing_images, testing_labels),
)

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save("image_classifier.model")  # saving the model

model = models.load_model("image_classifier.model")  # loading the model
