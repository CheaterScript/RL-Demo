import numpy as np
# import math


# def basic_sigmoid(x):
#     s = 1/(1+math.exp(-x))
#     return s


# def numpy_sigmoid(x):
#     x = np.array(x)
#     s = 1/(1+np.exp(-x))
#     return s


# print(basic_sigmoid(3))
# print(numpy_sigmoid([3, 2, 1]))


# def sigmoid_grad(x):
#     x = np.array(x)
#     s = numpy_sigmoid(x)
#     grad = s*(1-s)
#     return grad


# print(sigmoid_grad([1, 2, 3]))


# print(np.array([1,2,3])*np.array([[3],[2],[1]]))

def L1(y_hat, y):
    l1 = np.sum(np.abs(y - y_hat))
    return l1


print(L1(np.array([1, 2, 3]), np.array([3, 3, 3])))


def L2(y_hat, y):
    l1 = np.sum(np.abs(y - y_hat))
    return l1


print(L1(np.array([1, 2, 3]), np.array([3, 3, 3])))
