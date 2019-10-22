import numpy as np

class Neuron:
    def __init__(self, inputs_num, activation_func, d_activation_func):
        self.inputs_num = inputs_num
        self.weights = 2 * np.random.random(( self.inputs_num , 1)) - 1 # +1 for bias 
        self.learning_rate = 0.5
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
    
    def train(self, training_set, reference_set, iterations):
        for epoch in range(iterations):
            prediction = self.predict(training_set)
            error = reference_set - prediction
            learning_param = self.learning_rate * error * self.d_activation_func(prediction)
            weight_adjustment = np.dot(training_set.T, learning_param)
            self.weights += weight_adjustment

    def predict(self, input):
        return self.activation_func(np.dot(input, self.weights))

    

def neuron_heaviside_activate(state):
    return np.heaviside(state, 1)

def d_neuron_heaviside_activate(state):
    return 1