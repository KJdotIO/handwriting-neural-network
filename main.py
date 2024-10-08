import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load our model
model = tf.keras.models.load_model("handwriting.keras")

# transform our image: our MNIST dataset is grayscaled
# and 28 x 28 resolution. 
img = cv2.imread("one_test.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28)) 
img = np.invert(img)
img = img.reshape(1, 28, 28)

# this squishes each pixel value between 0 and 1
# this works better with what our inputs expect
# values between 0 and 1
img = img / 255.0

# generate our prediction ( 0-9 )
prediction = model.predict(img)
print(f"The number is probably a {np.argmax(prediction)}")

# use matplotlib to display our image in a canvas with
# the x and y axis going from 0 - 26
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()