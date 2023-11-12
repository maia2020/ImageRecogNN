import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

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

model = models.load_model("image_classifier.model")

# Now we will take random images from the internet to see if the model can identify them

img = cv.imread("deer.png")
# we need to convert the color scheme of the image to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)  # the index of the highest value in the array
print(f"Prediction is {class_names[index]}")

plt.show()
