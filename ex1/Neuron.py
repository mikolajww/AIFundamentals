import numpy as np

class Neuron:
    def __init__(self, inputs_num, activation_func, d_activation_func):
        self.eta = 0.05
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
        self.weights = np.zeros(inputs_num + 1)
    
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
                return 1 / (1 + np.exp(-state))
            return fn
        if name == "sin":
            def fn(state):
                return np.clip(np.sin(state), -1, 1)
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
    def d_activation_funcion(name):
        if name not in Neuron.available_functions():
            raise NotImplementedError
        if name == "step":
            def fn(state): 
                return 1
            return fn  
        if name == "sigmoid":
            def fn(state):
                return (1 / (1 + np.exp(-state))) * (1 - (1 / (1 + np.exp(-state))))
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

    def train(self, training_inputs, labels, iterations):
        for i in range(iterations):
            for input_point, label in zip(training_inputs, labels):
                error = label - self.predict(input_point)
                self.weights[1:] += self.eta * error * input_point
                self.weights[0] += self.eta * error

    def test_accuracy(self, inputs, labels):
        predictions = self.predict(inputs)
        errors = 0
        inputs_num = len(inputs)
        for prediction, label in zip(predictions, labels):
            if prediction != label:
                errors += abs(prediction - label)
        return 100 - (errors * 100/inputs_num)
                
    def predict(self, inputs):
        return self.activation_func(self.state(inputs))

    def state(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

