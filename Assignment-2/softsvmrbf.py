import math
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from pandas import DataFrame


def calc_kernal(x1: np.array, x2: np.array, sigma: float):
    norm = (np.linalg.norm(x1 - x2) ** 2)
    exp = - norm / (2. * sigma)
    return math.exp(exp)


def calc_gaussian_gram(trainX: np.array, sigma: float):
    m = trainX.shape[0]

    G = matrix(0., (m, m))
    for i in range(m):
        for j in range(m):
            G[i, j] = calc_kernal(trainX[i], trainX[j], sigma)

    return G


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmbf(l: float, sigma: float, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """

    m = trainX.shape[0]
    eye = spmatrix(1., range(m), range(m), (m, m))
    zeros = spmatrix(0., [], [], (m, m))

    G = calc_gaussian_gram(trainX, sigma)
    H = spdiag([(2 * l) * G, zeros])
    # check for positive definite
    helper_matrix = spmatrix(np.full(m, 1.e-5), range(m), range(m), (2 * m, 2 * m))
    H = H + helper_matrix

    #     |-----------------|
    #     |  G*2l  |   0    |
    # H = |--------|--------|    [2m x 2m]
    #     |   0    |   0    |
    #     |-----------------|

    u = (1 / m) * (np.append(np.zeros(m, dtype=float), np.ones(m, dtype=float)))
    u = matrix(list(u), (u.shape[0], 1))

    # u = [0,...,0,1/m,...,1/m]
    #        -m-       -m-

    x_list = []
    for i in range(m):
        x_list.append(trainy[i] * G[i, :])
    x = matrix(np.vstack(x_list))
    A = sparse([[x, zeros], [eye, eye]])

    #     |-------------------------|
    #     | -[y1*G1]-  |            |
    #     |    ...     |     Id     |
    #     | -[ym*Gm]-  |            |
    # A = |------------|------------|    [2m x 2m]
    #     |            |            |
    #     |      0     |     Id     |
    #     |            |            |
    #     |-------------------------|

    v = np.append(np.ones(m, dtype=float), np.zeros(m, dtype=float))
    v = matrix(list(v), (v.shape[0], 1))

    # v = [1,...,1,0,...,0]
    #        -m-     -m-

    sol = solvers.qp(H, u, -A, -v)
    alpha = sol['x']
    return np.asarray(alpha)[:m]


def predict_label(trainX: np.array, alpha: np.array, sigma: float, to_predict: np.array):
    sum: float = 0
    for i in range(trainX.shape[0]):
        sum += alpha[i][0] * calc_kernal(trainX[i], to_predict, sigma)
    return np.sign(sum)


def find_error(trainX: np.array, alpha: np.array, sigma: float, testX: np.array, testy: np.array):
    test_size = testX.shape[0]
    err = 0
    for i in range(test_size):
        if testy[i] != predict_label(trainX, alpha, sigma, testX[i]):
            err += 1
    err = float(err) / float(test_size)
    return err


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    alpha = softsvmbf(10, 0.1, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(alpha, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert alpha.shape[0] == m and alpha.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    d = trainX.shape[1]
    m = trainX.shape[0]

    # A)
    fig, ax = plt.subplots()
    positive_label_idx = np.where(trainy[:, 0] == 1)
    negative_label_idx = np.where(trainy[:, 0] == -1)
    ax.scatter(trainX[positive_label_idx, 0], trainX[positive_label_idx, 1], c='green', label="1")
    ax.scatter(trainX[negative_label_idx, 0], trainX[negative_label_idx, 1], c='red', label="-1")
    ax.legend()
    plt.show()

    # B) RBF
    folds = 5
    folds_x = np.split(trainX, folds)
    folds_y = np.split(trainy, folds)

    # find optimal l and sigma for softsvm-RBF
    err = np.infty
    opt_l = -1.
    opt_sigma = -1.

    err_per_sigma = []
    for sigma in (0.01, 0.5, 1.):
        err_per_l = []
        for lambd in (1., 10., 100.):
            print(f"\nStarted RBF to find alpha vals with \u03BB={lambd}, \u03C3={sigma} ...\n")
            curr_err = 0.
            for i in range(folds):
                # setup current fold
                curr_testX = folds_x[i]
                curr_testy = folds_y[i]

                if i == 0:
                    curr_trainX = np.concatenate(folds_x[1:])
                    curr_trainy = np.concatenate(folds_y[1:])
                elif i == folds - 1:
                    curr_trainX = np.concatenate(folds_x[:(folds - 1)])
                    curr_trainy = np.concatenate(folds_y[:(folds - 1)])
                else:
                    curr_trainX = np.concatenate((np.concatenate(folds_x[:i]), np.concatenate(folds_x[i + 1:])))
                    curr_trainy = np.concatenate((np.concatenate(folds_y[:i]), np.concatenate(folds_y[i + 1:])))

                print(f"Started fold number {i + 1} ...")
                alpha = softsvmbf(lambd, sigma, curr_trainX, curr_trainy)
                curr_err += find_error(curr_trainX, alpha, sigma, curr_testX, curr_testy)
            curr_err /= folds
            err_per_l.append(curr_err)
            err_per_l.append(curr_err)
            err_per_l.append(curr_err)
            print(f"Found curr error: {curr_err}")
            if curr_err < err:
                err = curr_err
                opt_l = lambd
                opt_sigma = sigma

        err_per_sigma.append(err_per_l)

    print("9 average validation error values for each of the pairs (\u03BB, \u03C3):")
    sigma_labels = ['          -0.1-', '          -0.5-', '          -1.0-']
    l_labels = ['-1-', '-10-', '-100-']
    print(DataFrame(err_per_sigma, columns=l_labels, index=sigma_labels))
    print("(columns for '\u03BB' values and rows for '\u03C3' values)")
    print(f"Best lambda value is: {opt_l}, and best sigma value is: {opt_sigma}")

    print(f"\n Running for the entire set with \u03BB={opt_l}, and \u03C3={opt_sigma}...")
    alpha = softsvmbf(opt_l, opt_sigma, trainX, trainy)
    err = find_error(trainX, alpha, opt_sigma, testX, testy)
    print(f" The error for the entire training set is: {err}")

     # find optimal l softsvm
    from softsvm import softsvm, err_on_set

    err = np.infty
    opt_l = -1

    err_per_l = []
    for lambd in (1., 10., 100.):
        print(f"\nStarted softsvm to find alpha vals with \u03BB={lambd} ...\n")
        curr_err = 0.
        for i in range(folds):
            # setup current fold
            curr_testX = folds_x[i]
            curr_testy = folds_y[i]

            if i == 0:
                curr_trainX = np.concatenate(folds_x[1:])
                curr_trainy = np.concatenate(folds_y[1:])
            elif i == folds-1:
                curr_trainX = np.concatenate(folds_x[:(folds-1)])
                curr_trainy = np.concatenate(folds_y[:(folds-1)])
            else:
                curr_trainX = np.concatenate((np.concatenate(folds_x[:i]), np.concatenate(folds_x[i + 1:])))
                curr_trainy = np.concatenate((np.concatenate(folds_y[:i]), np.concatenate(folds_y[i + 1:])))

            print(f"Started fold number {i + 1} ...")
            w = softsvm(lambd, curr_trainX, curr_trainy)
            curr_err += err_on_set(w, curr_testX, curr_testy)
        curr_err /= folds
        err_per_l.append(curr_err)
        if curr_err < err:
            err = curr_err
            opt_l = lambd

    print("\n3 average validation error values for each \u03BB:")
    l_labels = ['-1-', '-10-', '-100-']
    print(DataFrame(err_per_l, index=l_labels))
    print(f"best lambda value is: {opt_l}")

    print(f"\n Running for the entire set with l={opt_l}...")
    w = softsvm(opt_l, trainX, trainy)
    err = err_on_set(w, testX, testy)
    print(f" The error for the entire training set is: {err}")

    # D)
    print(f"\n Running for the entire set with \u03BB=100, and \u03C3=0.01...")
    alpha = softsvmbf(100, 0.01, trainX, trainy)
    grid = []
    for i in range(-50, 50):
        row = []
        for j in range(-50, 50):
            prediction = predict_label(trainX, alpha, 0.01, (float(i) / 5., float(j) / 5.))
            if prediction == -1:
                row.append([255, 0, 0])
            else:
                row.append([0, 0, 255])
        grid.append(row)

    plt.imshow(grid, extent=[-10, 10, -10, 10])
    plt.show()

    print(f"\n Running for the entire set with \u03BB=100, and \u03C3=0.5...")
    alpha = softsvmbf(100, 0.5, trainX, trainy)
    grid = []
    for i in range(-50, 50):
        row = []
        for j in range(-50, 50):
            prediction = predict_label(trainX, alpha, 0.5, (float(i) / 5., float(j) / 5.))
            if prediction == -1:
                row.append([255, 0, 0])
            else:
                row.append([0, 0, 255])
        grid.append(row)

    plt.imshow(grid, extent=[-10, 10, -10, 10])
    plt.show()

    print(f"\n Running for the entire set with \u03BB=100, and \u03C3=1.0...")
    alpha = softsvmbf(100, 1., trainX, trainy)
    grid = []
    for i in range(-50, 50):
        row = []
        for j in range(-50, 50):
            prediction = predict_label(trainX, alpha, 1., (float(i) / 5., float(j) / 5.))
            if prediction == -1:
                row.append([255, 0, 0])
            else:
                row.append([0, 0, 255])
        grid.append(row)

    plt.imshow(grid, extent=[-10, 10, -10, 10])
    plt.show()
