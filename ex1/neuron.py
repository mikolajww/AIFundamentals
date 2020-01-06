import numpy as np
import activation

class ActLayer:
    def __init__(self, input_size, output_size, f, f_prime):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = {}
        self.deltas = {}
        '''
        weights is a matrix representing weights and biases for all neurons in a layer
        weight:
        [
            neuron0_w = [w0, w1, ..., w_input_size],
            neuron1_w = [w0, w1, ..., w_input_size],
            ...,
            neuron_n_w = [w0, w1, ..., w_input_size]
        ]
        bias:
        [
            neuron0_b, neuron1_b, ..., neuron_n_b
        ]
        where n = output_size = number of neurons in a layer
        '''
        self.weights["weight"] = np.random.randn(input_size, output_size)
        self.weights["bias"] = np.random.randn(output_size)
        self.f = f
        self.f_prime = f_prime
    
    def reset(self):
        self.weights["weight"] = np.random.randn(self.input_size, self.output_size)
        self.weights["bias"] = np.random.randn(self.output_size)
        self.inputs = []
        self.state = []
        self.dektas = {}

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.state = inputs @ self.weights["weight"] + self.weights["bias"]
        return self.f(self.state)

    def backpropagate(self, accumulated_delta):
        local_delta = self.f_prime(self.state) * accumulated_delta
        self.deltas["bias"] = local_delta
        self.deltas["weight"] = self.inputs.T * local_delta
        return local_delta @ self.weights["weight"].T

class Tanh(ActLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size ,activation.activation_function("tanh"), activation.d_activation_funcion("tanh"))

class Step(ActLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, activation.activation_function("step"), activation.d_activation_funcion("step"))

class Sigmoid(ActLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, activation.activation_function("sigmoid"), activation.d_activation_funcion("sigmoid"))

class ReLU(ActLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, activation.activation_function("relu"), activation.d_activation_funcion("relu"))