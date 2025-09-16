import numpy as np

# Sigmoid
def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_sigmoid_der(x):
    return f_sigmoid(x)*(1 - f_sigmoid(x))

# Relu
def f_relu(x):
    return np.maximum(0, x)

def f_relu_der(x):
    return (x > 0).astype(float)

# Gelu
# It was a whim of mine to implement this function (Nicolas)
def f_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

def f_gelu_der(x):
    tanh_term = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * (x**3)))
    return 0.5 * (1 + tanh_term + x * (1 - tanh_term**2) * np.sqrt(2/np.pi) * (1 + 3*0.044715*(x**2)))

# Tanh
def f_tanh(x):
    return np.tanh(x)

def f_tanh_der(x):
    return 1 - (np.tanh(x))**2

class Layer():
    def __init__(self, in_nodes, out_nodes, activation="sigmoid"):
        if activation == "sigmoid":
            self.activation = f_sigmoid
            self.activation_derivative = f_sigmoid_der
        elif activation == "relu":
            self.activation = f_relu
            self.activation_derivative = f_relu_der
        elif activation == "gelu":
            self.activation = f_gelu
            self.activation_derivative = f_gelu_der
        elif activation == "tanh":
            self.activation = f_tanh
            self.activation_derivative = f_tanh_der
        else:
            print(f"Activation function {activation} not supported.")
            return
        
        self.in_size = in_nodes
        self.out_size = out_nodes

        self.w = np.random.randn(out_nodes, in_nodes)
        self.b = np.ones((out_nodes, 1))

        self.z = None
        self.a = None
        self.input_activations = None

        self.activation_name = activation
    
    def forward(self, x):
        self.input_activations = x
        self.z = np.matmul(self.w, x) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, delta, learning_rate=None):
        n_delta = np.dot(self.w.T, delta) * self.activation_derivative(self.z)

        grad_w = np.dot(delta, self.input_activations.T)
        grad_b = n_delta

        
        print(grad_w.shape)
        print(self.b.shape)
        print(grad_b.shape)

        if learning_rate is not None:
            self.w += -learning_rate * grad_w
            self.b += -learning_rate * grad_b

        return n_delta, grad_w, grad_b