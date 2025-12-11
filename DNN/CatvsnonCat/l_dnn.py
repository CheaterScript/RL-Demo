import numpy as np
import numpy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from network import *


#
# 实现一个两层网络，隐藏层有两个神经元，输出层有一个神经元
# #

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 5
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("y = " + str(train_set_y[:, index]) + ", it's a '" +
#       classes[np.squeeze(train_set_y.get()[:, index])].decode("utf-8") + "' picture.")

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

# GRADED FUNCTION: model


# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost

    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        # START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        # START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        # START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        # START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    costs_array = np.array(costs)
    # plt.plot(np.squeeze(costs_array).get())
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters


# d = model(train_set_x, train_set_y, test_set_x, test_set_y,
#           num_iterations=6770, learning_rate=0.001, print_cost=True)


layer_dims = (12288, 20, 7, 5, 1)

parameters = L_layer_model(train_set_x, train_set_y, layer_dims,
                           learning_rate=0.0075, num_iterations=2500, print_cost=True)
predict(train_set_x, train_set_y, parameters)
predict(test_set_x, test_set_y, parameters)

# # Example of a picture that was wrongly classified.
# index = 1
# # plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
# # plt.show()
# print(d["Y_prediction_test"][0, index])
# print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" +
#       classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")


# Plot learning curve (with costs)
# costs_array = np.array(d['costs'])
# costs = np.squeeze(costs_array)
# plt.plot(costs.get())
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()
