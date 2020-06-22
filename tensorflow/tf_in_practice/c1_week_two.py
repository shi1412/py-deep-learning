import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
print('Version of TensorFlow:',tf.__version__)

fasion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()
plt.imshow(train_images[0])
print(type(train_images))
print(type(train_labels))

print('Dimension of the labels:',train_labels.shape)
print('Dimension of the train images:', train_images.shape)
print('Dimension of the test images:', test_images.shape)

# Normalizing the data
train_images_norm = train_images / 255.0
test_images_norm = test_images / 255.0

# Design the model
# Adding more neurons may lead to more accurate but will hit the diminishing return very quickly
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), \
             tf.keras.layers.Dense(512, activation=tf.nn.relu), \
             tf.keras.layers.Dense(256, activation=tf.nn.relu),
             tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

"""
Sequential: That defines a SEQUENCE of layers in the neural network

Flatten: turns the 28*28 metric into a 1 dimensional set.

Dense: Adds a layer of neurons

Each layer of neurons need an activation function to tell them what to do. There's lots of options, but just use these for now.

Relu effectively: means "If X>0 return X, else return 0" -- so what it does is it only passes values 0 or greater to the next layer in the network.

Softmax: takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], 
it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!
"""
# Define a callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.4):
            print("\nReached 60% accuracy so cancelling training")
            self.model.stop_training = True
            
callback = myCallback()

# Build the model
model.compile(optimizer = tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(train_images_norm, train_labels, epochs = 5, callbacks=[callback])

# Test the model
model.evaluate(test_images_norm, test_labels)

classification = model.predict(test_images)
# These numbers are a probability that the value being classified is the corresponding value
print(classification[0])
print(test_labels[0])
