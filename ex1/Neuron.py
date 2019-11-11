import numpy as np

class Neuron:
    def __init__(self, inputs_num, activation_func, d_activation_func):
        self.eta = 0.25
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
        self.inputs_num = inputs_num
        self.weights = np.random.rand(inputs_num + 1)
    
    @staticmethod
    def available_functions():
        return ["step", "sigmoid", "sin", "tanh", "sign", "relu", "lrelu"]
    
    @staticmethod
    def activation_function(name):
        if name not in Neuron.available_functions():
            raise NotImplementedError
        if name == "step":
            def fn(state): 
                return np.heaviside(state, 1)
            return fn  
        if name == "sigmoid":
            def fn(state):
                beta = -1
                return 1 / (1 + np.exp(beta * state))
            return fn
        if name == "sin":
            def fn(state):
                return np.sin(state)
            return fn
        if name == "tanh":
            def fn(state):
                return np.tanh(state)
            return fn
        if name == "sign":
            def fn(state):
                return np.sign(state)
            return fn
        if name == "relu":
            def fn(state):
                return np.clip(state * (state > 0), -1, 1)
            return fn
        if name == "lrelu":
            def fn(state):
                return np.clip(np.where(state > 0, state, 0.01 * state), -1, 1)
            return fn
        else:
            raise NotImplementedError    

    @staticmethod
    def d_activation_function(name):
        if name not in Neuron.available_functions():
            raise NotImplementedError
        if name == "step":
            def fn(state): 
                return 1
            return fn  
        if name == "sigmoid":
            def fn(state):
                beta = -1
                return (1 / (1 + np.exp(beta * state))) * (1 - (1 / (1 + np.exp(beta * state))))
            return fn
        if name == "sin":
            def fn(state):
                return np.cos(state)
            return fn
        if name == "tanh":
            def fn(state):
                return 1.0 - np.tanh(state)**2
            return fn
        if name == "sign":
            def fn(state):
                return 1
            return fn
        if name == "relu":
            def fn(state):
                return np.where(state > 0, 1.0, 0.0)
            return fn
        if name == "lrelu":
            def fn(state):
                return np.where(state > 0, 1.0, 0.01)
            return fn
        else:
            raise NotImplementedError

    def reset_weights(self):
        self.weights = np.random.rand(self.inputs_num + 1)

    def train(self, training_inputs, labels, iterations):
        training_inputs, labels = unison_shuffled_copies(training_inputs, labels)
        eta = self.eta
        # print(f"Initial weights: {self.weights}")
        for i in range(iterations):
            eta -= 0.001 * i
            errors = 0
            for input_point, label in zip(training_inputs, labels):
                error = label - self.predict(input_point)
                #if(error != 0): print(f"Error (iteration {i}) : {error}")
                self.weights[1:] += eta * error * self.d_activation_func(self.state(input_point)) * input_point
                self.weights[0] += eta * self.d_activation_func(self.state(input_point)) * error
                errors += abs(error)
            # print(f"Weights after {i} iteration: {self.weights}")
        # print(f"Errors: {errors}")

    def predict(self, inputs):
        return self.activation_func(self.state(inputs))

    def state(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]