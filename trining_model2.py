from newFNN import newFNN
import numpy as np
import matplotlib.pyplot as plt
# Get the data from data_loading
from data_loading import train_images, train_labels, validation_images, validation_labels, test_images, test_labels

myNetwork = newFNN(in_nodes=784)
myNetwork.create_layer(out_nodes=16,activation="sigmoid")
myNetwork.create_layer(out_nodes=10,activation="sigmoid")

# Training parameters
training_data = train_images[:1000]
training_labels = train_labels[:1000]
batch_size = 10
epochs = 5
learning_rate = .01
# Train the model
myNetwork.train_SGD(training_data, training_labels, batch_size, epochs, learning_rate)
print(train_images[0].shape)
output = myNetwork.forward(train_images[0].reshape(-1, 1))
print(f"After SGD, output of the model has shape: {output.shape}\nExplicitly, the output is: \n{output.reshape(1, -1)}")

print(myNetwork.layers[0].b.shape)
print(myNetwork.layers[1].b.shape)
print(myNetwork.layers[2].b.shape)