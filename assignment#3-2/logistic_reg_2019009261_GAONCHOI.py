# 2019009261 Gaon Choi
# Practice 3-2
import argparse
import random
from copy import deepcopy as dc

import numpy as np
import time
import math
import matplotlib.pyplot as plt
import statistics

thr = 0.5   # threshold
eps = 1e-15

## Settings
# The number of nodes in each layer
XX = 1; HH = 2; YY = 1

# test case
test_case1 = [(10, 1000, 5000), (100, 1000, 5000), (10000, 1000, 5000)]
test_case2 = [(10000, 1000, 10), (10000, 1000, 100), (10000, 1000, 5000)]

def generate_dataset(m, n):
    # Generate m train samples, n test samples

    ## generate train samples
    x_train = []; y_train = []
    for i in range(m):
        rand = random.uniform(0, 2*np.pi)
        new_data = math.cos(rand)
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
        new_data = math.cos(rand)
        # rand = random.uniform(0, 360)
        # new_data = math.sin(math.radians(rand))
        x_test.append([rand])
        if new_data > 0:
            y_test.append(1)
        else:
            y_test.append(0)

    return x_train, y_train, x_test, y_test


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def elementwise_sigmoid(x, w, b):
    return sigmoid(w * x + b)

def vectorwise_sigmoid(X, W, b):
    return sigmoid(np.dot(W, X) + b)

def cross_entropy_loss(y, Y):
    # y: expected value
    # Y: actual value
    return -(Y * np.log(y + eps) + (1 - Y) * np.log(1 - y + eps))

def cross_entropy_derivative(y_, y):
    return (y / -(y_ + eps)) + ((1 - y) / (1 - y_ + eps))

def y_data_normalize(y):
    y[y >= thr] = 1
    y[y < thr] = 0
    return y

def vectorwise_learning(x_train, y_train, x_test, y_test, k, w1, b1, w2, b2, lr):
    # data format : list -> np.array
    x_train = np.array(x_train, dtype=float).T
    y_train = np.array(y_train, dtype=float)
    x_test = np.array(x_test, dtype=float).T
    y_test = np.array(y_test, dtype=float)
    w1 = np.array(w1, dtype=float)
    w2 = np.array(w2, dtype=float)
    b1 = np.array(b1, dtype=float)
    b2 = np.array(b2, dtype=float)

    # data size
    train_size = len(x_train.T)
    test_size = len(x_test.T)

    prev_time = time.time()
    cost = []
    for _ in range(k):
        # forward propagation
        z_ = np.dot(w1, x_train) + b1
        a_ = sigmoid(z_)
        aa_ = np.dot(w2, a_) + b2
        y = sigmoid(aa_)

        # backward propagation
        train_cost = cross_entropy_loss(y, y_train).sum()
        train_cost = train_cost / train_size

        ## Output layer
        dZ_2 = y - y_train
        dW_2 = (1 / train_size) * np.dot(dZ_2, a_.T)
        dB2 = (1 / train_size) * np.sum(dZ_2, axis=1)

        ## Hidden layer
        W2TdZ2 = np.dot(w2.T, dZ_2) / (train_size)
        g1Z1 = cross_entropy_derivative(z_, y_train) / (train_size)
        dZ_1 = W2TdZ2 * g1Z1
        dW_1 = (1 / train_size) * np.dot(dZ_1, x_train.T)
        dB1 = (1 / train_size) * np.sum(dZ_1, axis=1, keepdims=True)

        # make the expected value normalized
        y = y_data_normalize(y)

        # calculate the accuracy
        train_acc = np.sum(y == y_train)
        train_acc = (train_acc / train_size) * 100

        # update weights
        w1 = np.subtract(w1, lr*dW_1)
        b1 = np.subtract(b1, lr*dB1)
        w2 = np.subtract(w2, lr * dW_2)
        b2 = np.subtract(b2, lr * dB2)

        a_ = vectorwise_sigmoid(x_test, w1, b1)
        y_ = vectorwise_sigmoid(a_, w2, b2)
        test_cost = cross_entropy_loss(y_, y_test).sum()
        test_cost = test_cost / test_size
        cost.append(test_cost)

        if (_ + 1) % 1000 == 0: print("# {iter} | W={w1} {w2} | B={b1} {b2} | {tc}"
                                 .format(iter=_+1, w1=w1, w2=w2, b1=b1, b2=b2, tc=test_cost))

    elapsed_time = time.time() - prev_time

    # evaluation
    ## train data
    a_ = vectorwise_sigmoid(x_train, w1, b1)
    y_ = vectorwise_sigmoid(a_, w2, b2)
    y_ = y_data_normalize(y_)
    # calculate the accuracy
    train_acc = np.sum(y_ == y_train)
    train_acc = (train_acc / train_size) * 100

    ## test data
    a_ = vectorwise_sigmoid(x_test, w1, b1)
    y_ = vectorwise_sigmoid(a_, w2, b2)
    y_ = y_data_normalize(y_)

    # calculate the accuracy
    test_acc = np.sum(y_ == y_test)
    test_acc = (test_acc / test_size) * 100

    print("alpha = {lr}\n"
          "TRAIN-ACC: {acc1}%\n"\
          "TEST-ACC : {acc2}%\n".format(lr=lr, acc1=train_acc, acc2=test_acc))
    return w1, b1, w2, b2, train_acc, test_acc, elapsed_time, cost

def print_result(model_):
    strr = "W1: {w1}\n" \
           "W2: {w2}\n" \
           "B1: {b1}\n" \
           "B2: {b2}\n" \
           "Train Accuracy: {train}\n" \
           "Test Accuracy: {test}\n" \
           "Elapsed Time: {time}".format(w1=model_[0], b1=model_[1], w2=model_[2], b2=model_[3],train=model_[4],
                                         test=model_[5], time=model_[6])
    print(strr)


def set_parser(parser):
    parser.add_argument('m', type=int, default=10000)
    parser.add_argument('n', type=int, default=1000)
    parser.add_argument('k', type=int, default=1000)
    parser.add_argument('a', type=float, default = 0.01)
    parser.add_argument('mode', type=str, choices=['time', 'W', 'weights', 'alpha', 'data', 'a', 'calc', 'test'])

def main():
    parser = argparse.ArgumentParser()
    set_parser(parser)
    args = parser.parse_args()

    # Xavier Initialization
    xavier_init = math.sqrt(2/(HH+1))
    w1 = np.random.normal(0, xavier_init, (HH, XX))
    b1 = np.random.normal(0, xavier_init, (HH, XX))

    w2 = np.random.normal(0, xavier_init, (YY, HH))
    b2 = np.random.normal(0, xavier_init, (1, 1))

    if args.mode == 'time':
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
        print('Training with Vector-wise Model..')
        model = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w1, b1, w2, b2, 0.01)
        print('Elapsed Time: ', model[4], '(s)', sep='')

    if args.mode == 'W':
        weights = []
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)
        B2 = []
        mean_train = []
        mean_test = []
        for _ in range(100):
            x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
            w1_ = dc(w1); b1_ = dc(b1); w2_ = dc(w2); b2_ = dc(b2)
            model = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w1_, b1_, w2_, b2_, args.a)
            ax1.plot(round(model[0][0][0], 5), round(model[0][1][0], 5), '.')
            ax2.plot(round(model[2][0][0], 3), round(model[2][0][1], 3), '.')
            ax3.plot(round(model[1][0][0], 5), round(model[1][1][0], 5), '.')
            B2.append(round(model[3][0][0], 3))
            ax1.set_xlabel('w1[0]'); ax1.set_ylabel('w1[1]')
            ax2.set_xlabel('w2[0]'); ax2.set_ylabel('w2[1]')
            ax3.set_xlabel('b1[0]'); ax3.set_ylabel('b1[1]')
            ax4.set_xlabel('b2'); ax4.set_ylabel('#')
            weights.append([model[0][0], model[1]])
            mean_train.append(model[4])
            mean_test.append(model[5])
            print(_)
            del w1_, w2_, b1_, b2_, model
        ax1.set_title('W1')
        ax2.set_title('W2')
        ax3.set_title('B1')
        ax4.set_title('B2')
        ax4.hist(B2, bins=25, color='red', edgecolor='black')
        fig.suptitle("Estimated Model Parameters: W, b")
        # weights = np.array(weights)
        # theta = np.polyfit(weights[:, 0], weights[:, 1], 1)
        # y_line = theta[1] + theta[0] * weights[:, 0]
        # plt.plot(weights[:, 0], y_line, 'r')
        print("Train Acc Mean: ", statistics.mean(mean_train))
        print("Test  Acc Mean: ", statistics.mean(mean_test))
        plt.show()

    if args.mode == 'weights':
        # generate dataset
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
        model = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w1, b1, w2, b2, args.a)
        print('Training with Vector-wise Model..')
        print("Estimated Parameters:")
        print("W1: ", model[0])
        print("B1: ", model[1])
        print("W2: ", model[2])
        print("B2: ", model[3])

    if args.mode == 'alpha':
        # generate dataset
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)

        # comparison by alpha value
        alpha = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 20]

        kk = list(range(1, args.k+1, 1))
        pp = list(); i = 0
        ans_box = list()
        color = ['red', 'green', 'yellow', 'gold', 'deepskyblue', 'dodgerblue', 'crimson', 'magenta', 'olive',
                'lawngreen', 'forestgreen', 'lime', 'springgreen', 'aquamarine', 'turquoise', 'cyan']
        alpha.reverse(); color.reverse()
        for a in alpha:
            w1_ = dc(w1); b1_ = dc(b1); w2_ = dc(w2); b2_ = dc(b2)
            ans = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w1_, b1_, w2_, b2_, a)
            ans_box.append(ans[0:6])
            pp.append(ans[7])
            print("drawing... {alpha}".format(alpha=str(a)))

            plt.plot(kk, pp[i], color[i], label=str(a))
            i += 1

            del w1_, b1_, w2_, b2_
        alpha.reverse();
        color.reverse(); ans_box.reverse()

        plt.yscale('log')
        plt.xlabel('# of iterations (K)')
        plt.ylabel('Test Set Cost')
        plt.title('K - Test Cost Plot')
        plt.legend()
        plt.show()
        fig = plt.figure()
        figure_box = []
        theta = np.arange(0, 2 * np.pi + np.pi / 2, step=(np.pi / 2))

        for i in range(16):
            print("drawing.. {i}".format(i=i))
            figure_box.append(fig.add_subplot(4, 4, i+1))
            figure_box[i].plot(x_test, y_test, 'o', color='gainsboro', label='data')
            X = np.linspace(0, 2 * np.pi, 200)
            W1 = ans_box[i][0]; W2 = ans_box[i][2]
            B1 = ans_box[i][1]; B2 = ans_box[i][3]
            print(W1.shape, X.shape, B1.shape)
            X = np.reshape(X, (200,-1))
            Y = sigmoid(W1 @ X.T + B1)
            Y = sigmoid(W2 @ Y + B2)
            Y = Y.T
            print(X.shape)
            figure_box[i].plot(X, Y, color=color[i], label='model')
            figure_box[i].set_title("?? = {a}".format(a=alpha[i]))
            figure_box[i].set_xticks(theta, ['0', '??/2', '??', '3??/2', '2??'])
            figure_box[i].grid(True, linestyle='--')
            figure_box[i].legend(loc='upper right')
            figure_box[i].text(0.01, 0.01, 'ACC : {acc}%'.format(acc=round(ans_box[i][5], 2)), fontsize='x-small')
        fig.suptitle("Model and Dataset Visualization")
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
        ans = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w1, b1, w2, b2, args.a)

        kk = list(range(1, args.k + 1, 1))
        pp = list()
        pp.append(ans[7])
        print_result(ans)
        plt.plot(kk, pp[0], 'red', label=str(args.a))
        plt.yscale('log')
        plt.xlabel('# of iterations (K)')
        plt.ylabel('Test Set Cost')
        plt.title('K - Test Cost Plot')
        plt.legend()
        plt.show()

    if args.mode == 'calc':
        x_train, y_train, x_test, y_test = generate_dataset(args.m, args.n)
        result = []
        result_ = []
        for i in range(100):
            if (i+1) % 10 == 0: print(i+1)
            w1_ = w1; b1_ = b1; w2_ = w2; b2_ = b2
            ans = vectorwise_learning(x_train, y_train, x_test, y_test, args.k, w1_, b1_, w2_, b2_, 1)
            result.append(ans[2])
            result_.append(ans[3])
        print('Training: ', np.mean(result))
        print('Validation: ', np.mean(result_))

    if args.mode == 'test':
        for m, n, k in test_case1:
            print("-------<TEST>-------")
            print("M = {m}, N = {n}, K = {k}".format(m=m, n=n, k=k))
            x_train, y_train, x_test, y_test = generate_dataset(m, n)
            ans = vectorwise_learning(x_train, y_train, x_test, y_test, k, w1, b1, w2, b2, args.a)

            print_result(ans)
            del ans

        del x_train, y_train, x_test, y_test

        x_train, y_train, x_test, y_test = generate_dataset(10000, 1000)
        for m, n, k in test_case2:
            print("-------<TEST>-------")
            print("M = {m}, N = {n}, K = {k}".format(m=m, n=n, k=k))
            ans = vectorwise_learning(x_train, y_train, x_test, y_test, k, w1, b1, w2, b2, args.a)

            print_result(ans)
            del ans


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