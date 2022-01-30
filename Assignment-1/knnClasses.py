import numpy as np
from scipy.spatial import distance


class Sample:
    def __init__(self, x: np.array, y: int):
        self.X = x
        self.Y = y


class Norms:
    def __init__(self, k: int, to_check: np.array):
        self.k = k
        self.to_check = to_check
        # two list will hold k tuples of knn_object and their norms
        self.knn_array = np.empty(k, dtype=Sample)
        self.norm_array = np.full(k, np.inf)
        self.maxInd = 0

    def compare_update(self, sample: Sample):
        norm = distance.euclidean(self.to_check, sample.X)
        if self.norm_array[self.maxInd] > norm:
            self.knn_array[self.maxInd] = sample
            self.norm_array[self.maxInd] = norm
            self.update_max_ind()

    def update_max_ind(self):
        new_max_ind = 0
        new_max_norm = 0
        for i in range(self.k):
            if self.norm_array[i] > new_max_norm:
                new_max_norm = self.norm_array[i]
                new_max_ind = i
        self.maxInd = new_max_ind


class Classifier:
    def __init__(self, k: int):
        self.k = k
        self.m = 0
        self.training_db = []

    def push(self, sample: Sample):
        self.training_db.append(sample)
        self.m += 1

    def k_nearest_neighbors(self, to_check: np.array):
        # result vector ("knn") holds closest vectors
        norms = Norms(self.k, to_check)
        for sample in self.training_db:
            norms.compare_update(sample)
        return norms.knn_array

    def predict_digit(self, to_check: np.array):
        num_of_labels = 10
        knn_array = self.k_nearest_neighbors(to_check)
        label_count = np.zeros(num_of_labels)

        for neighbor in knn_array:
            label = int(neighbor.Y)
            label_count[label] += 1

        max_label_count = 0
        pred_label = 0
        for i in range(num_of_labels):
            if label_count[i] > max_label_count:
                max_label_count = label_count[i]
                pred_label = i

        return int(pred_label)
