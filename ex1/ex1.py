import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

mean_range = (-30, 30)
cov_range = (-5, 5)
colors = ["blue", "red"]
groups = {"blue": 2, "red": 3}
samples_per_group = {"blue": 500, "red": 1000}


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
        for j in range(0, samples_per_group):
            a, b = np.random.default_rng().multivariate_normal(means[i], covs[i]).T
            x.append(a)
            y.append(b)
    return x, y

def generate_and_plot_groups(groups, samples_per_group):
    ax.cla()
    for color in colors:
        x, y = random_samples_from_multivariate(groups[color], samples_per_group[color])
        ax.scatter(x, y, c=color)
    plt.show()


if __name__ == '__main__':
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    plt.autoscale(enable=True, axis='both', tight=True)
    red_groups = plt.axes([0.15, 0.17, 0.32, 0.05])
    red_samples_per_group = plt.axes([0.15, 0.11, 0.32, 0.05])
    blue_groups = plt.axes([0.63, 0.17, 0.32, 0.05])
    blue_samples_per_group = plt.axes([0.63, 0.11, 0.32, 0.05])
    generate = plt.axes([0.15, 0.05, 0.32, 0.05])

    red_groups_tb = TextBox(red_groups, 'Red groups', initial=str(groups["red"]))
    red_groups_tb.label.set_wrap(True)
    red_samples_per_group_tb = TextBox(red_samples_per_group, 'Red samples\nper group', initial=str(samples_per_group["red"]))
    red_samples_per_group_tb.label.set_wrap(True)

    blue_groups_tb = TextBox(blue_groups, 'Blue groups', initial=str(groups["blue"]))
    blue_groups_tb.label.set_wrap(True)
    blue_samples_per_group_tb = TextBox(blue_samples_per_group, 'Blue samples\nper group', initial=str(samples_per_group["blue"]))
    blue_samples_per_group_tb.label.set_wrap(True)

    generate_button = Button(generate, 'Generate random data')

    generate_button.on_clicked(lambda _: generate_and_plot_groups({"red": int(red_groups_tb.text), "blue": int(blue_groups_tb.text)},
                                                                  {"red": int(red_samples_per_group_tb.text), "blue": int(blue_samples_per_group_tb.text)}))

    #CLI Input

    # for color in colors:
    #     groups[color] = int(input(f"Enter number of {color} groups >>"))
    #     samples_per_group[color] = []
    #     for i in range(0, groups[color]):
    #         samples_per_group[color].append(int(input(f"Enter number of samples per {color} group {i + 1} >>")))
    generate_and_plot_groups(groups, samples_per_group)
