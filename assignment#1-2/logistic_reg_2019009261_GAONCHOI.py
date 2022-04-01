# 2019009261 Gaon Choi
# Practice 1-2
import argparse
import random
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from statistics import mean

thr = 0.5   # threshold
eps = 1e-15

def generate_dataset(m, n):
    # Generate m train samples, n test samples

    ## generate train samples
    x_train = []; y_train = []
    for i in range(m):
        rand = random.uniform(0, 2*np.pi)
        new_data = math.sin(rand)
        # rand = random.uniform(0, 360)
        # new_data = math.sin(math.radians(rand))
        x_train.append([rand])
        if new_data > 0:
            y_train.append(1)
        else:
            y_train.append(0)

    ## generate test samples
    x_test = []; y_test = []
    for i in range(n):
        rand = random.uniform(0, 2*np.pi)
        new_data = math.sin(rand)
        # rand = random.uniform(0, 360)
        # new_data = math.sin(math.radians(rand))
        x_test.append([rand])
        if new_data > 0:
            y_test.append(1)
        else:
            y_test.append(0)

    return x_train, y_train, x_test, y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def elementwise_sigmoid(x, w, b):
    return sigmoid(w * x + b)

def vectorwise_sigmoid(X, W, b):
    return sigmoid(np.dot(W, X) + b)

def cross_entropy_loss(y, Y):
    # y: expected value
    # Y: actual value
    return -(Y * np.log(y + eps) + (1 - Y) * np.log(1 - y + eps))

def comparison_test(y, Y):
    global thr
    if (y >= thr and Y == 1) or (y < thr and Y == 0):
        return True
    else:
        return False

def y_data_normalize(y, flag=False):
    if not flag:
        y[y >= thr] = 1
        y[y < thr] = 0
    else:
        for i in range(len(y)):
            y[i] = 1 if (y[i] >= thr) else 0
    return y

def elementwise_learning(x_train, y_train, x_test, y_test, k, w, b, lr):
    # data size
    train_size = len(x_train)
    test_size = len(x_test)

    prev_time = time.time()
    for _ in range(k):
        # cost
        train_cost = 0
        test_cost = 0

        # accuracy
        train_acc = 0
        test_acc = 0

        # gradient
        dw1 = 0; dw2 = 0; db = 0

        for i in range(train_size):
            x = x_train[i]; Y = y_train[i]
            y = elementwise_sigmoid(x, w, b)
            train_cost += cross_entropy_loss(y, Y)

            dz = y - Y
            dw1 += x[0] * dz; dw2 += x[1] * dz
            db += dz

            if comparison_test(y, Y):
                train_acc += 1
        # calculate cost
        train_cost = train_cost / train_size

        # calculate accuracy
        train_acc = train_acc / train_size * 100
        # normalize
        dw1 = dw1 / train_size
        dw2 = dw2 / train_size
        db = db / train_size

        # update weights
        w[0] -= lr * dw1
        w[1] -= lr * dw2
        b -= lr * db

    elapsed_time = time.time() - prev_time

    # evaluation
    ## train data
    y_ = [0 for i in range(train_size)]
    for i in range(train_size):
        y_[i] = elementwise_sigmoid(x_train[i], w, b)
    y_ = y_data_normalize(y_, True)

    train_acc = 0
    # calculate the accuracy
    for i in range(train_size):
        if y_[i] == y_train[i]:
            train_acc += 1
    train_acc = (train_acc / train_size) * 100

    ## test data
    y_ = [0 for i in range(test_size)]
    for i in range(test_size):
        y_[i] = elementwise_sigmoid(x_test[i], w, b)
    y_ = y_data_normalize(y_, True)

    test_acc = 0
    # calculate the accuracy
    for i in range(test_size):
        if y_[i] == y_test[i]:
            test_acc += 1
    test_acc = (test_acc / test_size) * 100

    return w, b, train_acc, test_acc, elapsed_time


def vectorwise_learning(x_train, y_train, x_test, y_test, k, w, b, lr):
    # data format : list -> np.array
    x_train = np.array(x_train).T
    y_train = np.array(y_train)
    x_test = np.array(x_test).T
    y_test = np.array(y_test)
    w = np.array(w); b = np.array(b)

    # data size
    train_size = len(x_train.T)
    test_size = len(x_test.T)

    prev_time = time.time()
    cost = []
    for _ in range(k):
        y = vectorwise_sigmoid(x_train, w, b)
        train_cost = cross_entropy_loss(y, y_train).sum()
        train_cost = train_cost / train_size

        A = y - y_train
        dW = np.dot(x_train, A)
        dB = A.sum()
        dW /= train_size
        dB /= train_size

        # make the expected value normalized
        y = y_data_normalize(y)

        # calculate the accuracy
        train_acc = np.sum(y == y_train)
        train_acc = (train_acc / train_size) * 100

        # update weights
        w -= lr * dW
        b -= lr * dB

        y_ = vectorwise_sigmoid(x_test, w, b)
        test_cost = cross_entropy_loss(y_, y_test).sum()
        test_cost = test_cost / test_size
        cost.append(test_cost)

    elapsed_time = time.time() - prev_time

    # evaluation
    ## train data
    y_ = vectorwise_sigmoid(x_train, w, b)
    y_ = y_data_normalize(y_)
    # calculate the accuracy
    train_acc = np.sum(y_ == y_train)
    train_acc = (train_acc / train_size) * 100

    ## test data
    y_ = vectorwise_sigmoid(x_test, w, b)
    y_ = y_data_normalize(y_)
    # calculate the accuracy
    test_acc = np.sum(y_ == y_test)
    test_acc = (test_acc / test_size) * 100

    return w, b, train_acc, test_acc, elapsed_time, cost

def set_parser(parser):
    parser.add_argument('m', type=int, default=10000)
    parser.add_argument('n', type=int, default=1000)
    parser.add_argument('k', type=int, default=1000)
    parser.add_argument('a', type=float, default = 0.01)
    parser.add_argument('mode', type=str, choices=['time', 'W', 'weights', 'alpha', 'data', 'a', 'calc'])

def main():
    parser = argparse.ArgumentParser()
    set_parser(parser)
    args = parser.parse_args()

    r1 = random.uniform(-1, 1)
    r2 = random.uniform(-1, 1)
    # r3 = random.uniform(-1, 1)

    # weights for element-wise learning
    w_elementwise = np.array([r1])
    b_elementwise = r2

    # weights for vector-wise learning
    w_vectorize = np.array([r1])
    b_vectorize = r2

    if args.mode == 'time':
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
        print('Training with Element-wise Model..')
        model = elementwise_learning(x_train, y_train, x_test, y_test, args.k, w_elementwise, b_elementwise, 0.01)
        print('Elapsed Time: ', model[4], '(s)', sep='')

        print('Training with Vector-wise Model..')
        model = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w_vectorize, b_vectorize, 0.01)
        print('Elapsed Time: ', model[4], '(s)', sep='')

    if args.mode == 'W':
        weights = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for _ in range(400):
            x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
            tmp1 = w_vectorize
            tmp2 = b_vectorize
            model = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, tmp1, tmp2, args.a)
            plt.plot(model[0][0], model[1], '.')
            weights.append([model[0][0], model[1]])
            print(_)
        plt.xlabel('w')
        plt.ylabel('b')
        weights = np.array(weights)
        theta = np.polyfit(weights[:, 0], weights[:, 1], 1)
        y_line = theta[1] + theta[0] * weights[:, 0]
        plt.title('Estimated Parameters: w, b')
        plt.plot(weights[:, 0], y_line, 'r')
        plt.show()

    if args.mode == 'weights':
        # generate dataset
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
        model = elementwise_learning(x_train, y_train, x_test, y_test, args.k, w_elementwise, b_elementwise, 0.01)
        print('Training with Element-wise Model..')
        print("Estimated Parameters:")
        print("W: ", model[0])
        print("B: ", model[1])
        model = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w_vectorize, b_vectorize, 0.01)
        print('Training with Vector-wise Model..')
        print("Estimated Parameters:")
        print("W: ", model[0])
        print("B: ", model[1])

    if args.mode == 'alpha':
        # generate dataset
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)

        # comparison by alpha value
        alpha = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 20]

        kk = list(range(1, args.k+1, 1))
        pp = list(); i = 0
        color = ['red', 'green', 'yellow', 'gold', 'deepskyblue', 'dodgerblue', 'crimson', 'magenta', 'olive',
                'lawngreen', 'forestgreen', 'lime', 'springgreen', 'aquamarine', 'turquoise']
        for a in alpha:
            ans = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w_vectorize, b_vectorize, a)
            pp.append(ans[5])
            print("drawing... {alpha}".format(alpha=str(a)))
            print(ans[0:4])
            plt.plot(kk, pp[i], color[i], label=str(a))
            i += 1
        plt.yscale('log')
        plt.xlabel('# of iterations (K)')
        plt.ylabel('Test Set Cost')
        plt.title('K - Test Cost Plot')
        plt.legend()
        plt.show()

    if args.mode == 'data':
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
        for _ in range(args.m):
            if y_train[_]:
                plt.plot(x_train[_], y_train[_], '|', color='red')
            else:
                plt.plot(x_train[_], y_train[_], '|', color='blue')

        plt.title('The Distribution of Data')
        plt.show()

    if args.mode == 'a':
        # generate dataset
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
        ans = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w_vectorize, b_vectorize, args.a)
        print(ans[3])
        kk = list(range(1, args.k + 1, 1))
        pp = list()
        pp.append(ans[5])
        plt.plot(kk, pp[0], 'red', label=str(args.a))
        plt.yscale('log')
        plt.xlabel('# of iterations (K)')
        plt.ylabel('Test Set Cost')
        plt.title('K - Test Cost Plot')
        plt.legend()
        plt.show()
        print(ans[0:4])
    if args.mode == 'calc':
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
        w_vectorize = np.array([r1])
        b_vectorize = r2
        result = []
        result_ = []
        for i in range(100):
            if (i+1) % 10 == 0: print(i+1)
            tmp1 = w_vectorize; tmp2 = b_vectorize
            ans = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, tmp1, tmp2, 1)
            result.append(ans[2])
            result_.append(ans[3])
        print('Training: ', mean(result))
        print('Validation: ', mean(result_))

if __name__ == '__main__':
    # manual --------------------------------------------------------------#
    # python logistic_reg_2019009261_GAONCHOI.py [m] [n] [k] [mode]        #
    # [mode] = 'time', 'W', 'weights', 'alpha'                             #
    ## 'time'    --> elapsed time comparison                               #
    ## 'W'       --> w0, w1 distribution plot                              #
    ## 'weights' --> W, b                                                  #
    ## 'alpha'   --> cost value comparison by the value of alpha           #
    ########################################################################
    main()