import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from losses import f_mse, f_mse_der, f_mse2, f_mse2_der

# Do NOT use

class newFNN():
    def __init__(self, in_nodes=None, loss = "mse2"):
        if loss == "mse":
            self.loss = f_mse
            self.loss_derivative = f_mse_der
        elif loss == "mse2":
            self.loss = f_mse2
            self.loss_derivative = f_mse2_der
        else:
            print(f"Loss function {loss} not accepted.")
            return
        if in_nodes == None:
            print(f"You must initialize the FNN with a number of nodes.")
            return
        
        self.output_size = in_nodes

        self.layers = []

    def create_layer(self, out_nodes, activation="sigmoid"):
        in_nodes = self.output_size
        self.output_size = out_nodes
        self.layers.append(Layer(in_nodes, out_nodes, activation))

    def forward_propagation(self, data):
        a = data
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def backward_propagation(self, X, y, learning_rate):
        # === Forward ===
        output = self.forward_propagation(X)

        grads_w = [None] * len(self.layers)
        grads_b = [None] * len(self.layers)

        delta = (output - y) * self.layers[-1].activation_derivative(self.layers[-1].z)
        grads_w[-1] = np.dot(delta, self.layers[-1].input_activations.T)
        grads_b[-1] = np.sum(delta, axis=1, keepdims=True)

        # === Backward ===
        for l in reversed(range(len(self.layers) - 1)):

            delta, grad_w, grad_b = self.layers[l].backward(delta, learning_rate)
            grads_w[l] = grad_w
            grads_b[l] = grad_b

        return grads_w, grads_b
    
    def update_params(self, grads_w, grads_b, lr=0.01):
        for i, layer in enumerate(self.layers):
            layer.w -= lr * grads_w[i]
            layer.b -= lr * grads_b[i]

    # Taken from the last update, thanks
    def train_SGD(self, training_data, training_labels, batch_size, epochs, learning_rate,
                  test_data=None, test_labels=None):

        n = training_data.shape[1]

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            print("Data shape:", training_data.shape)
            print("Labels shape:", training_labels.shape)

            training_data = training_data.T 
            training_labels = training_labels.T
            training_data = training_data[:, perm]
            training_labels = training_labels[:, perm]

            # Mini-batches
            for k in range(0, n, batch_size):
                X_batch = training_data[:, k:k+batch_size]
                y_batch = training_labels[:, k:k+batch_size]

                grads_w, grads_b = self.backward_propagation(X_batch, y_batch, learning_rate)

            print(f"Epoch {epoch+1}/{epochs}")
        self.plot_accuracy(epochs, batch_size, learning_rate)

    def plot_accuracy(self, epochs, batch_size, learning_rate):
        #x: epochs
        #y: accuracy of predictions per epoch
        x = np.arange(1, len(self.accuracy) + 1)
        plt.plot(x, self.accuracy, marker="o")

        plt.title(f"epochs: {epochs}, batch size: {batch_size}, learning rate: {learning_rate}")
        
        #Plot y between 0 and 100%
        plt.ylim(0, 101)
        plt.yticks(range(0, 101, 10))
        
        #x ticks at each epoch
        plt.xticks(x)

        # grid
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        
        plt.show()
    
    def vector_to_label(self, vec):
        return int(np.argmax(vec))
    
    def evaluate(self, test_data, test_labels, verbose=False):
        outputs = np.array([self.forward_propagation(sample) for sample in test_data])
        int_outputs = np.array([self.vector_to_label(vec) for vec in outputs])
        
        if verbose:
            correct_bools = int_outputs == test_labels
            total_correct = np.sum(correct_bools)
            self.accuracy.append(total_correct*100/len(correct_bools))
            print(f"Performance on test data: {total_correct}/{len(correct_bools)}, {total_correct*100/len(correct_bools):.2f}% acc")
        
        return outputs