import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


#
# 实现一个两层网络，隐藏层有两个神经元，输出层有一个神经元
# #

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 5
# plt.imshow(train_set_x_orig[index])
# plt.show()
print("y = " + str(train_set_y[:, index]) + ", it's a '" +
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig[0].shape[0]

# print("Number of training examples: m_train = " + str(m_train))
# print("Number of testing examples: m_test = " + str(m_test))
# print("Height/Width of each image: num_px = " + str(num_px))
# print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print("train_set_x shape: " + str(train_set_x_orig.shape))
# print("train_set_y shape: " + str(train_set_y.shape))
# print("test_set_x shape: " + str(test_set_x_orig.shape))
# print("test_set_y shape: " + str(test_set_y.shape))


train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    # START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-z))
    ### END CODE HERE ###

    return s


def sigmoid_grad(z):

    grad = sigmoid(z) * (1-sigmoid(z))
    return grad


def relu(z):
    return np.maximum(z, 0)


def relu_grad(z):
    return np.where(z > 0, 1.0, 0.0)


print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))


# GRADED FUNCTION: initialize_with_zeros

def initialize_W(dims):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    # START CODE HERE ### (≈ 1 line of code)
    W = np.random.randn(dims[0], dims[1])
    ### END CODE HERE ###

    assert (W.shape == dims)

    return W


def initialize_B(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    # START CODE HERE ### (≈ 1 line of code)
    B = np.zeros((dim, 1))
    ### END CODE HERE ###

    assert (B.shape == (dim, 1))
    return B


dims = (2, 2)
W = initialize_W(dims)
print("w = " + str(W))
B = initialize_B(dims[0])
print("b = " + str(B))

# GRADED FUNCTION: propagate
print("# GRADED FUNCTION: propagate")


def propagate(W1, B1, W2, B2, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    # START CODE HERE ### (≈ 2 lines of code)
    Z1 = W1@X+B1  # (2,n) * (n, batchs) = (2, batchs) + (2, 1) = (2, batchs)
    A1 = sigmoid(Z1)  # (2, batchs)
    Z2 = W2@A1 + B2  # (1,2) * (2,batchs) = (1,batchs) + (1,1) = (1,batchs)
    A2 = sigmoid(Z2)            # compute activation
    cost = -1/m*np.sum(Y * np.log(A2)+(1-Y) *
                       np.log(1-A2))    # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    # START CODE HERE ### (≈ 2 lines of code)
    dZ2 = A2 - Y  # (1, batchs) - (1, batchs) = (1, batchs)
    dW2 = 1/m * dZ2@A1.T  # (1,batchs) * (batchs, 2) = (1, 2)
    dB2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)  # (1,1)
    # (1,2).T = (2,1) * (1,batchs) = (2, batchs) x (2, batchs) = (2,batchs)
    dZ1 = W2.T @ dZ2 * sigmoid_grad(Z1)
    dW1 = 1/m * dZ1 @ X.T  # (2,batchs) * (batchs,n) = (2, n)
    dB1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)  # (2,1)
    ### END CODE HERE ###

    assert (dW1.shape == W1.shape)
    # print(dB1.dtype, B1.shape)
    # assert (dB1.dtype == B1.shape)
    assert (dW2.shape == W2.shape)
    # assert (dB2.dtype == B2.shape)
    # assert (dB2.dtype == B2.shape)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dW1": dW1,
             "dB1": dB1,
             "dW2": dW2,
             "dB2": dB2,
             }

    return grads, cost


# w, b, X, Y = np.array([[1], [2]]), 2, np.array(
#     [[1, 2], [3, 4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))
# print("cost = " + str(cost))


# GRADED FUNCTION: optimize

def optimize(W1, B1, W2, B2, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(W1, B1, W2, B2, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dW1 = grads["dW1"]
        dB1 = grads["dB1"]
        dW2 = grads["dW2"]
        dB2 = grads["dB2"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        W1 = W1 - learning_rate * dW1
        B1 = B1 - learning_rate * dB1
        W2 = W2 - learning_rate * dW2
        B2 = B2 - learning_rate * dB2
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"W1": W1,
              "B1": B1,
              "W2": W2,
              "B2": B2
              }

    grads = {"dW1": dW1,
             "dB1": dB1,
             "dW2": dW2,
             "dB2": dB2,
             }

    return params, grads, costs


# params, grads, costs = optimize(
#     w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

# print("w = " + str(params["w"]))
# print("b = " + str(params["b"]))
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))


# GRADED FUNCTION: predict

def predict(W1, B1, W2, B2, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    # w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    # START CODE HERE ### (≈ 1 line of code)
    Z1 = W1@X+B1  # (2,n) * (n, m) + (2,1) = (2,m)
    A1 = sigmoid(Z1)  # (2, m)
    z2 = W2 @ A1 + B2  # (1, 2) * (2,m) + (1,1) = (1,m)
    a2 = sigmoid(z2)  # (1,m)
    ### END CODE HERE ###

    for i in range(a2.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        # START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[0, i] = int(1) if a2[0, i] > 0.5 else int(0)
        ### END CODE HERE ###

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# print("predictions = " + str(predict(w, b, X)))


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###

    # initialize parameters with zeros (≈ 1 line of code)
    print(X_train.shape[0])
    W1 = initialize_W((2, X_train.shape[0]))
    B1 = initialize_B(2)
    W2 = initialize_W((1, 2))
    B2 = initialize_B(1)
    # w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(
        W1, B1, W2, B2, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(W1, B1, W2, B2, X_test)
    Y_prediction_train = predict(W1, B1, W2, B2, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "W1": W1,
         "B1": B1,
         "W2": W2,
         "B2": B2,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=100000, learning_rate=0.03, print_cost=True)


# # Example of a picture that was wrongly classified.
# index = 1
# # plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
# # plt.show()
# print(d["Y_prediction_test"][0, index])
# print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" +
#       classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
