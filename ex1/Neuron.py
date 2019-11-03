import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Neuron:
    def __init__(self, no_of_inputs, activation_func, d_activation_func):
        self.eta = 0.05
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
        self.weights = np.zeros(no_of_inputs + 1)
           
    def train(self, training_inputs, labels, iterations, plot):
        zipped = list(zip(training_inputs, labels))
        line, = plt.plot([], [], '-k')
        def animate(i):
            input_point, label = zipped[i]
            error = label - self.predict(input_point)
            self.weights[1:] += self.eta * error * input_point
            self.weights[0] += self.eta * error
            x = np.linspace(-50, 50, 100)
            y = (-self.weights[1]/self.weights[2]) * x + (-self.weights[0]/self.weights[2])
            # print((-self.weights[1]/self.weights[2]), (-self.weights[0]/self.weights[2]))
            line.set_data(x, y)
            return line,
        anim = animation.FuncAnimation(plot, animate, frames=len(labels), interval=0.5, repeat=False)

                


    def predict(self, inputs):
        return self.activation_func(self.state(inputs))

    def state(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

def neuron_heaviside_activate(state):
    return np.heaviside(state, 1)

def d_neuron_heaviside_activate(state):
    return 1

