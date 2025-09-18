####### Try loading, training, and evaluating a model
from network import FNN
from newFNN import newFNN
import numpy as np
import matplotlib.pyplot as plt
# Get the data from data_loading
from data_loading import train_images, train_labels, validation_images, validation_labels, test_images, test_labels
#print(train_images.shape[0], validation_images.shape[0])

# Creates a feedforward network with 3 layers, input (28x28), hidden (16), output(10)
network = FNN([784,16, 10])

# Training parameters
batch_size = 10
epochs = 5
learning_rate = 3
# Train the model
network.train_SGD(train_images, train_labels, batch_size, epochs, learning_rate, test_images, test_labels)
print(train_images[0].shape)
# Evaluate the model
print("Model after training:")
_outputs = network.evaluate(test_images, test_labels, verbose=True)


#Attack on the network by modifying the input data slightly
success = 0
amount = 10000

attack_images = np.zeros((amount, 784))
# Evaluate (amount) of attack images
for i in range(amount):
    data_index = i
    x = test_images[data_index].reshape(784, 1)
    y = np.zeros((10,1))
    y[test_labels[data_index]] = 1

    original_prediction = np.argmax(network.forward(x))

    grad_input = network.input_gradient(x,y)
    epsilon = 0.3
    x_attack = x + epsilon * np.sign(grad_input)
    attack_prediction = np.argmax(network.forward(x_attack))
    if attack_prediction != original_prediction:
        success += 1

    attack_images[i] = x_attack.flatten()
print(f"Attack success rate: {100*success/amount}%")

attack_labels = test_labels[:amount]

#Train the network on the attack data
epochs = 15
learning_rate = 2
network.train_SGD(attack_images, attack_labels, batch_size, epochs, learning_rate)

#Perform the attack again, with the network trained on the attack data.
success = 0
amount = 10000

for i in range(amount):
    data_index = i
    x = test_images[data_index].reshape(784, 1)
    y = np.zeros((10,1))
    y[test_labels[data_index]] = 1

    original_prediction = np.argmax(network.forward(x))

    grad_input = network.input_gradient(x,y)
    epsilon = 0.3
    x_attack = x + epsilon * np.sign(grad_input)
    attack_prediction = np.argmax(network.forward(x_attack))
    if attack_prediction != original_prediction:
        success += 1

print(f"Attack success rate: {100*success/amount}%")

##Plot results
#plt.figure(figsize=(10,6))
#plt.title(f"epochs: {epochs}, batch size: {batch_size}, learning rate: {learning_rate}, epsilon: {epsilon}")
#plt.subplot(1,2,1)
#plt.title(f"Original\nPrediction: {orig_pred}, Correct: {test_labels[data_index]}")
#plt.imshow(x.reshape(28,28), cmap='gray')
#plt.axis('off')
#plt.subplot(1,2,2)
#plt.title(f"Attack\nPrediction: {atk_pred}, Correct: {test_labels[data_index]}")
#plt.imshow(x_atk.reshape(28,28), cmap='gray')
#plt.axis('off')
#
#plt.show()