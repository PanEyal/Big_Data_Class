import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


def softsvm(l, trainX: np.array, trainy: np.array):
    """
    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m = trainX.shape[0]
    d = trainX.shape[1]

    v = np.append(np.ones(m), np.zeros(m))
    v = matrix(list(v), (v.shape[0], 1))

    # v = [1,...,1,0,...,0]
    #        -m-     -m-

    H = spmatrix(2 * l, range(d), range(d), (d+m, d+m))

    # H = Identity*(2*l) [d+m x d+m]

    u = np.append(np.zeros(d, dtype=float), np.full(m, 1./m, dtype=float))
    u = matrix(list(u), (u.shape[0], 1))

    # u = [0,...,0,1,...,1]
    #        -d-     -m-

    x_list = []
    for i in range(m):
        x_list.append(trainy[i] * trainX[i])
    x = matrix(np.vstack(x_list))

    eye = spmatrix(1, range(m), range(m), (m, m))
    zeros = spmatrix(0, [], [], (m, d))
    A = sparse([[x, zeros], [eye, eye]])

    #     |-------------------------|
    #     | -[y1*X1]-  |            |
    #     |    ...     |     Id     |
    #     | -[ym*Xm]-  |            |
    # A = |------------|------------|    [d+m x 2m]
    #     |            |            |
    #     |      0     |     Id     |
    #     |            |            |
    #     |-------------------------|

    sol = solvers.qp(H, u, G=-A, h=-v)
    w = np.array(sol['x'])[:d]
    return w


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


# Input - a list of random indexes in regards to the training data, and training data
# Output - a tuple of lists of examples and corresponding labels, with the indexes given in the input
def smpl_from_indx_lst(indx_lst, train_X: np.array, train_Y: np.array):
    tmp_lst_X = []
    tmp_lst_Y = []
    m = indx_lst.shape[0]
    for i in range(m):
        rnd_indx = indx_lst[i]
        tmp_lst_X.append(train_X[rnd_indx])
        tmp_lst_Y.append(train_Y[rnd_indx])
    tmp_lst_X = np.array(tmp_lst_X)
    tmp_lst_Y = np.array(tmp_lst_Y)
    return tmp_lst_X, tmp_lst_Y


def err_on_set(w: np.array, X_data: np.array, Y_data: np.array):
    test_size = X_data.shape[0]
    err_w = 0
    for i in range(test_size):
        pred = np.sign(X_data[i] @ w)
        if pred != Y_data[i]:
            err_w += 1
    err_w = float(err_w) / float(test_size)
    return err_w


def create_rnd_smple(m: int, num: int, X_train: np.array, Y_train: np.array):
    smpl_lst = []
    for i in range(num):
        rnd_indx_lst = np.random.randint(0, trainX.shape[0], m)
        smpl_lst.append(smpl_from_indx_lst(rnd_indx_lst, X_train, Y_train))
    return smpl_lst


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()

    # here you may add any code that uses the above functions to solve question 2
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m_1 = 100
    m_2 = 1000
    iter = 10
    num_lambdas = 10

    # run for l in {1, ... , 10}, for each 'l' run 'iter' times and average, then plot.
    X = range(1, num_lambdas+1)
    Y1_train = np.zeros(num_lambdas)
    Y1_test = np.zeros(num_lambdas)
    Y1_train_err = np.zeros((2, iter))
    Y1_test_err = np.zeros((2, iter))
    Y2_train = np.zeros(num_lambdas)
    Y2_test = np.zeros(num_lambdas)

    for l in X:
        # 2.1 /
        print(f"started for lambda = {l}")
        train_errors = np.zeros(iter)
        test_errors = np.zeros(iter)
        smpl_lst = create_rnd_smple(m_1, iter, trainX, trainy)
        for i in range(iter):
            curr_smpl = smpl_lst[i]
            w = softsvm(10 ** l, curr_smpl[0], curr_smpl[1])
            train_errors[i] = err_on_set(w, curr_smpl[0], curr_smpl[1])
            test_errors[i] = err_on_set(w, testX, testy)
        train_mean_err = float(sum(train_errors)) / float(iter)
        test_mean_err = float(sum(test_errors)) / float(iter)
        train_min_err = np.amin(train_errors)
        test_min_err = np.amin(test_errors)
        train_max_err = np.amax(train_errors)
        test_max_err = np.amax(test_errors)
        Y1_train[l - 1] = float(sum(train_errors)) / float(iter)
        Y1_test[l - 1] = float(sum(test_errors)) / float(iter)
        Y1_train_err[0][l - 1] = train_mean_err - train_min_err
        Y1_train_err[1][l - 1] = train_max_err - train_mean_err
        Y1_test_err[0][l - 1] = test_mean_err - test_min_err
        Y1_test_err[1][l - 1] = test_max_err - test_mean_err
        # / 2.1

        # 2.2 /
        smpl = create_rnd_smple(m_2, 1, trainX, trainy)[0]
        w = softsvm(10 ** l, smpl[0], smpl[1])
        Y2_train[l - 1] = err_on_set(w, smpl[0], smpl[1])
        Y2_test[l - 1] = err_on_set(w, testX, testy)
        print("hi")
        # / 2.2

    print("plotting...hehehe...")
    plt.plot(X, Y1_train, color='orange', marker="o", label="train results; m=100")
    plt.plot(X, Y1_test, color='green', marker="o", label="test results; m=100")
    plt.errorbar(X, Y1_train, Y1_train_err, fmt="none", ecolor='orange')
    plt.errorbar(X, Y1_test, Y1_test_err, fmt="none", ecolor='green')
    plt.scatter(X, Y2_train, color='blue', label="train results; m=1000")
    plt.scatter(X, Y2_test, color='purple', label="test results; m=1000")
    plt.xlabel("$log_{10}(\lambda)$")
    plt.ylabel("Mean Error")
    plt.title("Q2")
    plt.legend()
    plt.show()


