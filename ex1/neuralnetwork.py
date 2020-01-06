from neuron import *

def get_batch(training_set, reference_set, batch_size = 64, shuffle = True):
    starts = np.arange(0, len(training_set), batch_size)
    if shuffle:
        np.random.shuffle(starts)

    for start in starts:
        end = start + batch_size
        batch_training_set = training_set[start:end]
        batch_reference_set = reference_set[start:end]
        dict = {
            'training_set': batch_training_set, 
            'reference_set': batch_reference_set
        }
        yield dict

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def loss(self, predicted, actual):
        return np.sum((predicted - actual) ** 2)

    def delta(self, predicted, actual):
        return 2 * (predicted - actual)
    
    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward_pass(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

    def backpropagate(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backpropagate(delta)

    def train(self, training_set, reference_set, epochs = 500, verbose = False):
        eta = 0.1
        for epoch in range(epochs):
            loss = 0.0
            for input_point, label in zip(training_set, reference_set):
                prediction = self.forward_pass(np.array([input_point]))
                loss += self.loss(prediction, label)
                delta = self.delta(prediction, label)
                self.backpropagate(delta)
                for layer in self.layers:
                    for name, param in layer.weights.items():
                        d = layer.deltas[name].reshape(param.shape)
                        param -= eta * d
            eta = eta * 0.99
            if(verbose):
                print(epoch, loss)


