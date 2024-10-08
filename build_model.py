import numpy as np
import tensorflow as tf

# load our data
path = 'mnist.npz'
with np.load(path) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# normalise the data by scaling down between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# create the model
model = tf.keras.models.Sequential()
# input layer
model.add(tf.keras.Input(shape=(28,28)))

# this turns it into a flattened layer of 28 x 28 inputs
# into a 1D vector
model.add(tf.keras.layers.Flatten())

# using a ReLU as our neurones, and softmax for our outputs
# we will use two hidden layers of 128 neurones.
# and an output layer with 10 neurones for classification (0 - 9)
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#compile the model
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# evaluate the cost and accuracy of our untrained model
initial_cost, initial_accuracy = model.evaluate(x_test, y_test)

# print the cost and accuracy of our untrained model
print(f"Initial cost: {initial_cost}, Initial accuracy: {initial_accuracy}")

# train our model 3 times
# this might be a bit overkill for a NN recognising
# handwritten digits, but doesn't matter
model.fit(x_train, y_train, epochs=3, validation_split=0.1)

# evaluate the cost and accuracy of our trained model
final_cost, final_accuracy = model.evaluate(x_test, y_test)

# print the cost and accuracy of our trained model
print(f"Final cost: {final_cost}, Final accuracy: {final_accuracy}")

# save our model to root dir
model.save('handwriting.keras')