import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix


def z_sorce_normalize_arrays(arr_list: np.array) -> np.array:
    mean = np.mean(arr_list, axis=1)
    std = np.std(arr_list, axis=1)
    normalized_data = (arr_list - mean[:, np.newaxis]) / std[:, np.newaxis]
    return normalized_data


def set_env(seed, device='0'):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Metrics:
    def __init__(self):
        self.test_accuracy_list = []
        self.train_accuracy_list = []
        self.all_metrics_list = []

    def calculate_accuracy(self, true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        print('Current test accuracy is {}'.format(percentage(accuracy)))

    def test_calculate_accuracy(self, true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        self.test_accuracy_list.append(accuracy)
        return percentage(accuracy)

    def train_calculate_accuracy(self, true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        self.train_accuracy_list.append(accuracy)
        return percentage(accuracy)

    def test_average_accuracy(self):
        if len(self.test_accuracy_list) == 0:
            raise ValueError('test_accuracy_list length is zero.')
        return calculate_mean_list(self.test_accuracy_list)

    def train_average_accuracy(self):
        if len(self.train_accuracy_list) == 0:
            raise ValueError('train_accuracy_list length is zero.')
        return calculate_mean_list(self.train_accuracy_list)
    def calculate_all_metrics(self, true_labels, predicted_labels, average='weighted'):
        """
        accuracy, f1, sensitivity, precision, auc组成元组(accuracy, f1, sensitivity, precision, auc),存储于self.all_metrics_list中
        :param true_labels:
        :param predicted_labels:
        :param average:
        :return:
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average=average)
        sensitivity = recall_score(true_labels, predicted_labels, average=average)
        precision = precision_score(true_labels, predicted_labels, average=average)
        auc = roc_auc_score(true_labels, predicted_labels, average=average)
        specificity= self.specificity_score(true_labels, predicted_labels)
        self.all_metrics_list.append((accuracy, f1, sensitivity, precision,specificity, auc))
        print('Test accuracy_score: {}%'.format(percentage(accuracy)))
        print('Test f1_score: {}%'.format(percentage(f1)))
        print('Test recall_score: {}%'.format(percentage(sensitivity)))
        print('Test precision_score: {}%'.format(percentage(precision)))
        print('Test specificity_score: {}%'.format(percentage(specificity)))
        print('Test roc_auc_score: {}%'.format(percentage(auc)))
        return accuracy, f1, sensitivity, precision, auc

    def average_all_metrics(self):
        means = [sum(t) / len(t) for t in zip(*self.all_metrics_list)]
        std_devs = [((sum((x - mean) ** 2 for x in t) / len(t)) ** 0.5) for mean, t in zip(means, zip(*self.all_metrics_list))]

        average_list = list(map(lambda x: round(x*100, 2), means))
        average_std = list(map(lambda x: round(x*100, 2), std_devs))

        print(
            f'All average metrics: average-accuracy:{average_list[0]}±{average_std[0]},'
            f' average-f1-score:{average_list[1]}±{average_std[1]},'
            f' average-sensitivity:{average_list[2]}±{average_std[2]},'
            f' average-precision_score:{average_list[3]}±{average_std[3]},'
            f' average-specificity_score:{average_list[4]}±{average_std[4]},'
            f' average-roc_auc_score:{average_list[5]}±{average_std[5]}.')
        return average_list

    def accuracy_in_all_metrics(self):
        average_list = []
        for idx, (accuracy, f1, sensitivity, precision,specificity, auc) in enumerate(self.all_metrics_list):
            average_list.append(accuracy)
        return average_list

    def accuracy_average_in_all_metrics(self):
        return calculate_mean_list(self.accuracy_in_all_metrics())

    def specificity_score(self, true_labels, predicted_labels):
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        specificity = tn / (tn + fp)
        return specificity


def percentage(value, decimals=2):
    return round(100 * value, decimals)


def calculate_mean_list(lst, decimals=2, Percentage: bool = True):
    mean = np.mean(lst)
    if Percentage:
        mean *= 100
    return round(mean, decimals)
