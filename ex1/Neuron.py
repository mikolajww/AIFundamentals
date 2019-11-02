import numpy as np

class Neuron:
    def __init__(self, inputs_num, activation_func, d_activation_func):
        self.inputs_num = inputs_num + 1
        self.weights = np.random.rand(self.inputs_num) # +1 for bias [w0, w1, w2]
        self.learning_rate = 0.5
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
    def train(self, training_set, reference_set, iterations):
        # [[11111111], [i01 i0n], [i11, i1n]] - training set
        # [label_n * training_set_length]
        training_set = np.hstack(training_set, np.ones(len(reference_set)), 0)
        for epoch in range(iterations):
            for j in range(len(reference_set)):
                input_sample = np.array(training_set.T[j])
                prediction = self.predict(input_sample)
                error = reference_set[j] - prediction
                delta_weight = self.learning_rate * error * self.d_activation_func(self.state(input_sample)) * input_sample
                self.weights += delta_weight

    def predict(self, input):
        return self.activation_func(self.state(input))

    def state(self, input):
        s = self.weights @ input
        return s

def neuron_heaviside_activate(state):
    return np.heaviside(state, 1)

def d_neuron_heaviside_activate(state):
    return 1