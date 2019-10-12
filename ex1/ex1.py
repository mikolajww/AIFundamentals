import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

mean_range = (-50, 50)
cov_range = (-5, 5)


def rand_range(a, b):
    return (b - a) * np.random.random() + a


def rand_mean():
    return [rand_range(mean_range[0], mean_range[1]), rand_range(mean_range[0], mean_range[1])]


def rand_cov():
    a = rand_range(cov_range[0], cov_range[1])
    return [[rand_range(cov_range[0], cov_range[1]), a], [a, rand_range(cov_range[0], cov_range[1])]]


def random_samples_from_multivariate(groups, samples_per_group):
    means = []
    covs = []
    for i in range(0, groups):
        means.append(rand_mean())
        covs.append(rand_cov())
    x = []
    y = []
    for i in range(0, groups):
        for j in range(0, samples_per_group[i]):
            a, b = np.random.default_rng().multivariate_normal(means[i], covs[i]).T
            x.append(a)
            y.append(b)
    return x, y


if __name__ == '__main__':
    groups = {}
    samples_per_group = {}
    colors = ["blue", "red"]
    for color in colors:
        groups[color] = int(input(f"Enter number of {color} groups >>"))
        samples_per_group[color] = []
        for i in range(0, groups[color]):
            samples_per_group[color].append(int(input(f"Enter number of samples per {color} group {i + 1} >>")))
    for color in colors:
        x, y = random_samples_from_multivariate(groups[color], samples_per_group[color])
        plt.scatter(x, y, c=color)
    plt.show()
