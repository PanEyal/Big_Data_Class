from random import choice

import matplotlib.pyplot as plt
import numpy as np

import knnClasses


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """

    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def genall(x_list: list, y_list: list):
    """
    gensmallm unite a sample of labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
        :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """

    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])

    arranged_x = x[indices]
    arranged_y = y[indices]

    return arranged_x, arranged_y


def gen_corrupted(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels with 20% of corruption in data.
    if m == -1 then generate all samples

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    if (m != -1):
        rearranged_x = rearranged_x[:m]
        rearranged_y = rearranged_y[:m]

    for i in range(int(len(rearranged_y) * 0.2)):
        labels = y_list.copy()
        curr_label = int(rearranged_y[i])
        labels.remove(curr_label)

        rearranged_y[i] = choice(labels)

    return rearranged_x, rearranged_y


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """
    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """

    classifier = knnClasses.Classifier(k)
    for i in range(x_train.shape[0]):
        sample = knnClasses.Sample(x_train[i], y_train[i])
        classifier.push(sample)

    return classifier


def predictknn(classifier, x_test: np.array):
    """
    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """

    n = x_test.shape[0]
    prediction = np.empty((n, 1), dtype=int)
    for i in range(n):
        prediction[i] = classifier.predict_digit(x_test[i])
    return prediction


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def train(k, x_train_list, sample_size):
    x_train, y_train = gensmallm(x_train_list, [1, 3, 4, 6], sample_size)

    classifier = learnknn(k, x_train, y_train)

    return classifier


def train_with_corrupted(k, x_train_list, sample_size):
    x_train, y_train = gen_corrupted(x_train_list, [1, 3, 4, 6], sample_size)

    classifier = learnknn(k, x_train, y_train)

    return classifier


def findError(classifier: knnClasses.Classifier, x_test, y_test):
    preds = predictknn(classifier, x_test)
    y_test = np.reshape(y_test, (len(y_test), 1))
    # calculate the error
    return np.mean(y_test != preds)


def graph_per_sample_size(iter_num: int, k: int, sample_min: int, sample_max: int, sample_step: int):
    data = np.load('mnist_all.npz')

    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    x_test, y_test = genall([test1, test3, test4, test6], [1, 3, 4, 6])

    num_of_samples = int((sample_max - sample_min) / sample_step)
    training_sample_sizes = np.zeros(num_of_samples, dtype=int)
    mean_errors = np.zeros(num_of_samples)
    y_err = np.zeros((2, num_of_samples))
    errors_per_sample = np.zeros(iter_num)

    for i in range(num_of_samples):
        training_sample_sizes[i] = sample_min + (i * sample_step)
        print(f"started for sample size: {training_sample_sizes[i]}")
        tss = int(training_sample_sizes[i])

        for j in range(iter_num):
            classifier = train(k, [train1, train3, train4, train6], tss)
            errors_per_sample[j] = findError(classifier, x_test, y_test)

        mean_errors[i] = sum(errors_per_sample) / iter_num
        min_err = errors_per_sample[np.argmin(errors_per_sample)]
        max_err = errors_per_sample[np.argmax(errors_per_sample)]
        y_err[0][i] = mean_errors[i] - min_err
        y_err[1][i] = max_err - mean_errors[i]

    return training_sample_sizes, mean_errors, y_err


def graph_per_k(iter_num: int, k_min: int, k_max: int, sample_size: int):
    data = np.load('mnist_all.npz')

    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    x_test, y_test = genall([test1, test3, test4, test6], [1, 3, 4, 6])

    training_size = k_max - k_min + 1
    mean_errors = np.zeros(training_size)
    y_err = np.zeros((2, training_size))
    errors_per_sample = np.zeros(iter_num)
    training_k_sizes = np.zeros(training_size, dtype=int)

    for i in range(training_size):
        curr_k = k_min + i
        print(f"started for k: {curr_k}")
        training_k_sizes[i] = curr_k

        for j in range(iter_num):
            classifier = train(curr_k, [train1, train3, train4, train6], sample_size)
            errors_per_sample[j] = findError(classifier, x_test, y_test)

        mean_errors[i] = sum(errors_per_sample) / iter_num
        min_err = errors_per_sample[np.argmin(errors_per_sample)]
        max_err = errors_per_sample[np.argmax(errors_per_sample)]
        y_err[0][i] = mean_errors[i] - min_err
        y_err[1][i] = max_err - mean_errors[i]

    return training_k_sizes, mean_errors, y_err


def graph_per_k_with_corrupted(iter_num: int, k_min: int, k_max: int, sample_size: int):
    data = np.load('mnist_all.npz')

    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    x_test, y_test = gen_corrupted([test1, test3, test4, test6], [1, 3, 4, 6], -1)

    training_size = k_max - k_min + 1
    mean_errors = np.zeros(training_size)
    y_err = np.zeros((2, training_size))
    errors_per_sample = np.zeros(iter_num)
    training_k_sizes = np.zeros(training_size, dtype=int)

    for i in range(training_size):
        curr_k = k_min + i
        print(f"started for k: {curr_k}")
        training_k_sizes[i] = curr_k

        for j in range(iter_num):
            classifier = train_with_corrupted(curr_k, [train1, train3, train4, train6], sample_size)
            errors_per_sample[j] = findError(classifier, x_test, y_test)

        mean_errors[i] = sum(errors_per_sample) / iter_num
        min_err = errors_per_sample[np.argmin(errors_per_sample)]
        max_err = errors_per_sample[np.argmax(errors_per_sample)]
        y_err[0][i] = mean_errors[i] - min_err
        y_err[1][i] = max_err - mean_errors[i]

    return training_k_sizes, mean_errors, y_err


if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # question 2
    # a)
    iter_num = 10
    k = 1
    sample_min = 1
    sample_max = 21
    sample_step = 2
    sample_size1, mean_errors1, y_err1 = graph_per_sample_size(iter_num, k, sample_min, sample_max, sample_step)
    sample_min = 21
    sample_max = 110
    sample_step = 10
    sample_size2, mean_errors2, y_err2 = graph_per_sample_size(iter_num, k, sample_min, sample_max, sample_step)

    sample_size = np.concatenate((sample_size1, sample_size2))
    mean_errors = np.concatenate((mean_errors1, mean_errors2))
    y_err = np.hstack((y_err1, y_err2))

    plt.plot(sample_size, mean_errors, color='red', marker="o")
    plt.errorbar(sample_size, mean_errors, y_err, fmt="none", ecolor='blue')
    plt.xlabel("Training Sample Size")
    plt.ylabel("Mean Error")
    plt.title(f"Mean Error of KNN with k = {k}")
    plt.show()

    # e)
    iter_num = 10
    k_min = 1
    k_max = 11
    sample_size = 100
    k_sizes, mean_errors, y_err = graph_per_k(iter_num, k_min, k_max, sample_size)
    plt.plot(k_sizes, mean_errors, color='red', marker="o")
    plt.errorbar(k_sizes, mean_errors, y_err, fmt="none", ecolor='blue')
    plt.xlabel("K")
    plt.ylabel("Mean Error")
    plt.title(f"Mean Error of KNN per k with sample size = {sample_size}")
    plt.show()

    # f)
    iter_num = 10
    k_min = 1
    k_max = 11
    sample_size = 100
    k_sizes, mean_errors, y_err = graph_per_k_with_corrupted(iter_num, k_min, k_max, sample_size)
    plt.plot(k_sizes, mean_errors, color='red', marker="o")
    plt.errorbar(k_sizes, mean_errors, y_err, fmt="none", ecolor='blue')
    plt.xlabel("K")
    plt.ylabel("Mean Error")
    plt.title(f"Mean Error of KNN per k with sample size = {sample_size}")
    plt.show()
