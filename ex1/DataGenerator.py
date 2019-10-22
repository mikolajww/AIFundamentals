from typing import Dict
import numpy as np


def rand_range(a, b):
    return (b - a) * np.random.random() + a


def rand_mean(mean_range):
    return [rand_range(mean_range[0], mean_range[1]), rand_range(mean_range[0], mean_range[1])]


def rand_cov(cov_range):
    a = rand_range(cov_range[0], cov_range[1])
    return [[rand_range(cov_range[0], cov_range[1]), a], [a, rand_range(cov_range[0], cov_range[1])]]


class DataGenerator:
    def __init__(self, mean_range, covariance_range, class_label) -> None:
        self.mean_range = mean_range
        self.cov_range = covariance_range
        self.class_label = class_label

    def get_samples(self, groups, samples_per_group):
        means = []
        covs = []
        for i in range(0, groups):
            means.append(rand_mean(self.mean_range))
            covs.append(rand_cov(self.cov_range))
        x = []
        y = []
        for i in range(0, groups):
            a, b = np.random.multivariate_normal(means[i], covs[i], samples_per_group, check_valid='ignore').T
            x.append(a)
            y.append(b)
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        labels = [self.class_label for x in x]
        return [x, y], labels