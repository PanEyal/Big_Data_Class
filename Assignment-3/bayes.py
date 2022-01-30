import numpy as np
import matplotlib.pyplot as plt

def genall(x_list: list, y_list: list):
    """
    genall generates all samples along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])

    arranged_x = x[indices]
    arranged_y = y[indices]

    return arranged_x, arranged_y


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


def bayeslearn(x_train: np.array, y_train: np.array):
    """
    :param x_train: 2D numpy array of size (m, d) containing the the training set. The training samples should be binarized
    :param y_train: numpy array of size (m, 1) containing the labels of the training set
    :return: a triple (allpos, ppos, pneg) the estimated conditional probabilities to use in the Bayes predictor
    """
    samples_size = x_train.shape[0]
    features_size = x_train.shape[1]
    allpos = np.sum(y_train[y_train == 1.]) / samples_size
    ppos = np.full(features_size, 0.)
    pneg = np.full(features_size, 0.)

    for feature in range(features_size):
        feature_array = np.transpose(x_train[:, feature])

        if allpos != 0.:
            ppos[feature] = (np.sum(feature_array[y_train == 1.]) / samples_size) / allpos
            if ppos[feature] == 0.:
                ppos[feature] = float("nan")
        if allpos != 1.:
            pneg[feature] = (np.sum(feature_array[y_train != 1.]) / samples_size) / (1 - allpos)
            if pneg[feature] == 0.:
                pneg[feature] = float("nan")

    return allpos, ppos, pneg


def bayespredict(allpos: float, ppos: np.array, pneg: np.array, x_test: np.array):
    """
    :param allpos: scalar between 0 and 1, indicating the fraction of positive labels in the training sample
    :param ppos: numpy array of size (d, 1) containing the empirical plug-in estimate of the positive conditional probabilities
    :param pneg: numpy array of size (d, 1) containing the empirical plug-in estimate of the negative conditional probabilities
    :param x_test: numpy array of size (n, d) containing the test samples
    :return: numpy array of size (n, 1) containing the predicted labels of the test samples
    """
    samples_size = x_test.shape[0]
    feature_size = x_test.shape[1]

    y_predict = np.zeros(samples_size)

    for sample in range(samples_size):
        predict_pos = np.log(allpos)
        predict_neg = np.log(1 - allpos)
        for feature in range(feature_size):
            if not np.isnan(ppos[feature]) and not np.isnan(pneg[feature]):
                if x_test[sample, feature] == 1.:
                    predict_pos += np.log(ppos[feature])
                    predict_neg += np.log(pneg[feature])
                else:
                    predict_pos += np.log(1 - ppos[feature])
                    predict_neg += np.log(1 - pneg[feature])

        y_predict[sample] = 1. if predict_pos > predict_neg else -1.

    return y_predict.reshape((samples_size, 1))


def simple_test():
    # load sample data from question 2, digits 3 and 5 (this is just an example code, don't forget the other part of
    # the question)
    data = np.load('mnist_all.npz')

    train3 = data['train3']
    train5 = data['train5']

    test3 = data['test3']
    test5 = data['test5']

    m = 500
    n = 50
    d = train3.shape[1]

    x_train, y_train = gensmallm([train3, train5], [-1, 1], m)

    x_test, y_test = gensmallm([test3, test5], [-1, 1], n)
    y_test = y_test.reshape((n,1))

    # threshold the images (binarization)
    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    x_test = np.where(x_test > threshold, 1, 0)

    # run naive bayes algorithm
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    assert isinstance(ppos, np.ndarray) \
           and isinstance(pneg, np.ndarray), "ppos and pneg should be numpy arrays"

    assert 0 <= allpos <= 1, "allpos should be a float between 0 and 1"

    y_predict = bayespredict(allpos, ppos, pneg, x_test)

    assert isinstance(y_predict, np.ndarray), "The output of the function bayespredict should be numpy arrays"
    assert y_predict.shape == (n, 1), f"The output of bayespredict should be of size ({n}, 1)"

    print(f"Prediction error = {np.mean(y_test != y_predict)}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    threshold = 128
    
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train3 = data['train3']
    train5 = data['train5']

    test0 = data['test0']
    test1 = data['test1']
    test3 = data['test3']
    test5 = data['test5']

    x_0_1_test, y_0_1_test = genall([test0, test1], [-1, 1])
    x_0_1_test = np.where(x_0_1_test > threshold, 1, 0)
    y_0_1_test = y_0_1_test.reshape((y_0_1_test.shape[0], 1))

    x_3_5_test, y_3_5_test = genall([test3, test5], [-1, 1])
    x_3_5_test = np.where(x_3_5_test > threshold, 1, 0)
    y_3_5_test = y_3_5_test.reshape((y_3_5_test.shape[0], 1))

    # a.
    sample_sizes = np.arange(start=1000, stop=10001, step=1000)
    error_0_1 = []
    error_3_5 = []

    for sample_size in sample_sizes:
        print(f"Started with sample size={sample_size} on 0_1")
        x_0_1_train, y_0_1_train = gensmallm([train0, train1], [-1, 1], sample_size)
        x_0_1_train = np.where(x_0_1_train > threshold, 1, 0)

        # run naive bayes algorithm
        allpos, ppos, pneg = bayeslearn(x_0_1_train, y_0_1_train)
        y_0_1_predict = bayespredict(allpos, ppos, pneg, x_0_1_test)

        error_0_1.append(np.mean(y_0_1_test != y_0_1_predict))

        print(f"Started with sample size={sample_size} on 3_5")
        x_3_5_train, y_3_5_train = gensmallm([train3, train5], [-1, 1], sample_size)
        x_3_5_train = np.where(x_3_5_train > threshold, 1, 0)

        # run naive bayes algorithm
        allpos, ppos, pneg = bayeslearn(x_3_5_train, y_3_5_train)
        y_3_5_predict = bayespredict(allpos, ppos, pneg, x_3_5_test)

        error_3_5.append(np.mean(y_3_5_test != y_3_5_predict))

    plt.plot(sample_sizes, error_3_5 , color='red', marker="o", label="error on 3,5")
    plt.plot(sample_sizes, error_0_1 , color='blue', marker="o", label="error on 0,1")
    plt.xlabel("Sample Size")
    plt.ylabel("Error")
    plt.title("Q2 b.")
    plt.legend()
    plt.show()

    # c.
    x_0_1_train, y_0_1_train = gensmallm([train0, train1], [-1, 1], 10000)
    x_0_1_train = np.where(x_0_1_train > threshold, 1, 0)

    # run naive bayes algorithm
    allpos, ppos, pneg = bayeslearn(x_0_1_train, y_0_1_train)

    plt.title("ppos: one")
    fixed_ppos = np.where(np.isnan(ppos), 0., ppos)
    plt.imshow(fixed_ppos.reshape((28,28)), cmap='hot')
    plt.show()
    plt.title("pneg: zero")
    fixed_pneg = np.where(np.isnan(pneg), 0., pneg)
    plt.imshow(fixed_pneg.reshape((28,28)), cmap='hot')
    plt.show()

    # d.
    x_0_1_train, y_0_1_train = gensmallm([train0, train1], [-1, 1], 10000)
    x_0_1_train = np.where(x_0_1_train > threshold, 1, 0)

    allpos, ppos, pneg = bayeslearn(x_0_1_train, y_0_1_train)
    y_0_1_predict = bayespredict(allpos, ppos, pneg, x_0_1_test)
    y_0_1_altered_predict = bayespredict(0.75, ppos, pneg, x_0_1_test)

    changed_to_1 = 0.
    changed_to_minus_1 = 0.
    for i in range(y_0_1_predict.shape[0]):
        if y_0_1_predict[i] == -1. and y_0_1_altered_predict[i] == 1.:
            changed_to_1 += 1
        if y_0_1_predict[i] == 1. and y_0_1_altered_predict[i] == -1.:
            changed_to_minus_1 += 1

    changed_to_1 /= y_0_1_predict.shape[0]
    changed_to_minus_1 /= y_0_1_predict.shape[0]

    print(f"percent of the 0_1 set that their label changed from -1 to 1: {changed_to_1}")
    print(f"percent of the 0_1 set that their label changed from 1 to -1: {changed_to_minus_1}")

    x_3_5_train, y_3_5_train = gensmallm([train3, train5], [-1, 1], 10000)
    x_3_5_train = np.where(x_3_5_train > threshold, 1, 0)

    allpos, ppos, pneg = bayeslearn(x_3_5_train, y_3_5_train)
    y_3_5_predict = bayespredict(allpos, ppos, pneg, x_3_5_test)
    y_3_5_altered_predict = bayespredict(0.75, ppos, pneg, x_3_5_test)

    changed_to_1 = 0.
    changed_to_minus_1 = 0.
    for i in range(y_3_5_predict.shape[0]):
        if y_3_5_predict[i] == -1. and y_3_5_altered_predict[i] == 1.:
            changed_to_1 += 1
        if y_3_5_predict[i] == 1. and y_3_5_altered_predict[i] == -1.:
            changed_to_minus_1 += 1

    changed_to_1 /= y_3_5_predict.shape[0]
    changed_to_minus_1 /= y_3_5_predict.shape[0]

    print(f"percent of the 3_5 set that their label changed from -1 to 1: {changed_to_1}")
    print(f"percent of the 3_5 set that their label changed from 1 to -1: {changed_to_minus_1}")

