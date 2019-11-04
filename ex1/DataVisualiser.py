import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.colors import ListedColormap
import numpy as np
from DataGenerator import DataGenerator
import Neuron as n

class DataVisualiser:
    def __init__(self) -> None:
        self.mean_range = (-30, 30)
        self.cov_range = (-5, 5)
        self.colors = ["blue", "red"]
        self.colors_to_label = {"red": 0, "blue": 1}
        self.groups = {"blue": 1, "red": 1}
        self.samples_per_group = {"blue": 100, "red": 100}
        self.generators = {
            "blue": DataGenerator(self.mean_range, self.cov_range, self.colors_to_label["blue"]),
            "red": DataGenerator(self.mean_range, self.cov_range, self.colors_to_label["red"])
        }
        self.data = {"blue": {}, "red": {}}
        self.neuron = n.Neuron(2, n.Neuron.activation_function("lrelu"), n.Neuron.d_activation_funcion("lrelu"))
        self.fig, self.ax = plt.subplots()
        self.xlim = []
        self.ylim = []
        self.colorbar = None
        plt.subplots_adjust(bottom=0.3)
        plt.autoscale(enable=True, axis='both', tight=True)
        red_groups = plt.axes([0.15, 0.17, 0.32, 0.05])
        red_samples_per_group = plt.axes([0.15, 0.11, 0.32, 0.05])
        blue_groups = plt.axes([0.63, 0.17, 0.32, 0.05])
        blue_samples_per_group = plt.axes([0.63, 0.11, 0.32, 0.05])
        generate = plt.axes([0.15, 0.05, 0.32, 0.05])
        train = plt.axes([0.63, 0.05, 0.32, 0.05])

        red_groups_tb = TextBox(red_groups, 'Red groups', initial=str(self.groups["red"]))
        red_groups_tb.label.set_wrap(True)
        red_samples_per_group_tb = TextBox(red_samples_per_group, 'Red samples\nper group',
                                           initial=str(self.samples_per_group["red"]))
        red_samples_per_group_tb.label.set_wrap(True)

        blue_groups_tb = TextBox(blue_groups, 'Blue groups', initial=str(self.groups["blue"]))
        blue_groups_tb.label.set_wrap(True)
        blue_samples_per_group_tb = TextBox(blue_samples_per_group, 'Blue samples\nper group',
                                            initial=str(self.samples_per_group["blue"]))
        blue_samples_per_group_tb.label.set_wrap(True)

        generate_button = Button(generate, 'Generate random data', hovercolor='lightblue')
        train_button = Button(train, 'Train', hovercolor='lightgreen')
        train_button.on_clicked(lambda _: self.train())
        generate_button.on_clicked(
            lambda _: self.update_and_plot_groups({"red": int(red_groups_tb.text), "blue": int(blue_groups_tb.text)},
                                                  {"red": int(red_samples_per_group_tb.text),
                                                   "blue": int(blue_samples_per_group_tb.text)}))
        plt.show()
       

    def update_and_plot_groups(self, groups, samples_per_group):
        self.groups = groups
        self.samples_per_group = samples_per_group
        self.generate_and_plot_groups()

    def generate_and_plot_groups(self):
        self.ax.cla()
        for color in self.colors:
            samples, labels = self.generators[color].get_samples(self.groups[color], self.samples_per_group[color])
            self.data[color]["samples"] = samples
            self.data[color]["labels"] = labels
            self.ax.scatter(samples[0], samples[1], c=color)
        self.ylim = self.ax.get_ylim()
        self.xlim = self.ax.get_xlim()
    
    def merge_to_tuples(self, xs, ys):
        tuples = []
        for x, y in zip(xs, ys):
            tuples.append([x, y])
        return tuples

    def train(self):
        self.ax.cla()
        for color in self.colors:
            self.ax.scatter(self.data[color]["samples"][0], self.data[color]["samples"][1], c=color)
        self.ylim = self.ax.get_ylim()
        self.xlim = self.ax.get_xlim()
        training_set = [[], []]
        reference_set = []
        for color in self.colors:
            training_set[0].extend(self.data[color]["samples"][0])
            training_set[1].extend(self.data[color]["samples"][1])
            reference_set.extend(self.data[color]["labels"])
        training_set = np.array(training_set).T
        self.neuron.train(np.array(training_set), np.array(reference_set), 10)
        dim = np.arange(self.mean_range[0] + 3*self.cov_range[0], self.mean_range[1] + 3*self.cov_range[1], 0.1)
        xx, yy = np.meshgrid(dim, dim)
        decision_region = self.neuron.predict(np.array([xx.ravel(), yy.ravel()]).T)
        decision_region = decision_region.reshape(xx.shape)
        contour = self.ax.contourf(xx, yy, decision_region, alpha=0.5, cmap='RdBu')
        accuracy = self.neuron.test_accuracy(np.array(training_set), np.array(reference_set))
        self.ax.set_title(f"Accuracy: {accuracy}%")
        if self.colorbar is None:
            self.colorbar = plt.colorbar(contour, ax=self.ax)
        # x = np.linspace(-50, 50, 100)
        # y = (-self.neuron.weights[1]/self.neuron.weights[2]) * x + (-self.neuron.weights[0]/self.neuron.weights[2])
        # self.ax.plot(x, y, '-g')
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)


visualizer = DataVisualiser()