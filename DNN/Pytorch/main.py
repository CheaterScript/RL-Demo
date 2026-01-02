import torch
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot

y_hat = torch.tensor(36)
y = torch.tensor(39)

loss = (y - y_hat)**2

print(loss.item())


# GRADED FUNCTION: linear_function
def linear_function():

    torch.manual_seed(1)
    X = torch.randn((3, 1))
    W = torch.randn((4, 3))
    b = torch.zeros((4, 1))
    Y = W@X + b

    return Y


print("result = " + str(linear_function()))

# GRADED FUNCTION: sigmoid


def sigmoid(z):
    x = torch.tensor(z)
    return torch.sigmoid(x)


print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(12) = " + str(sigmoid(12)))

# GRADED FUNCTION: cost


def cost(logits, labels):
    z = logits
    y = torch.tensor(labels, dtype=torch.float32)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        input=z, target=y, reduction='none')

    return loss


logits = sigmoid([[0.2, 0.4, 0.7, 0.9]])
cost = cost(logits, [[0, 0, 1, 1]])
print("cost = " + str(cost))

# GRADED FUNCTION: one_hot_matrix


def one_hot_matrix(labels, C):
    C = torch.tensor(C)
    one_hot_matrix = torch.nn.functional.one_hot(labels, C)

    return one_hot_matrix


labels = torch.tensor([1, 2, 3, 0, 2, 1])
one_hot = one_hot_matrix(labels, C=4)
print("one_hot = " + str(one_hot))


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
plt.show()
print("y = " + str(np.squeeze(Y_train_orig[:, index])))

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print("number of training examples = " + str(X_train.shape[1]))
print("number of test examples = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# GRADED FUNCTION: initialize_parameters


def initialize_parameters():
    torch.manual_seed(1)

    W1 = torch.nn.init.xavier_uniform(torch.empty(
        25, 12288))
    b1 = torch.zeros((25, 1))
    W2 = torch.nn.init.xavier_uniform(torch.empty(
        12, 25))
    b2 = torch.zeros((12, 1))
    W3 = torch.nn.init.xavier_uniform(torch.empty(
        6, 12))
    b3 = torch.zeros((6, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }

    for name, tensor in parameters.items():
        parameters[name] = tensor.requires_grad_(True)

    return parameters


parameters = initialize_parameters()
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# GRADED FUNCTION: forward_propagation
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # Z1 = np.dot(W1, X) + b1
    Z1 = W1@X + b1
    # A1 = relu(Z1)
    A1 = torch.nn.functional.relu(Z1)
    # Z2 = np.dot(W2, a1) + b2
    Z2 = W2@A1 + b2
    # A2 = relu(Z2)
    A2 = torch.nn.functional.relu(Z2)
    # Z3 = np.dot(W3,Z2) + b3
    Z3 = W3@A2 + b3

    return Z3


X = torch.zeros((12288, 1))
parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters)
print("Z3 = " + str(Z3))

# GRADED FUNCTION: compute_cost

# !!!!!!!!!!
def compute_cost(Z3, Y,debug=False):
    # PyTorch 期望的形状是 (batch_size, num_classes)，所以需要转置
    # Z3 形状: (6, num_examples) -> 转置为 (num_examples, 6)
    logits = Z3.t()
    
    # Y 形状: (6, num_examples) -> 转置为 (num_examples, 6)
    labels = Y.t()
    
    # 如果是 one-hot 编码，转换为类别索引
    if labels.dim() == 2 and labels.shape[1] > 1:
        # 假设 Y 是 one-hot 编码，转换为类别索引
        labels = torch.argmax(labels, dim=1)
    
    # 计算交叉熵损失（内部包含 softmax）
    # 注意：CrossEntropyLoss 期望 logits（未经过 softmax）
    cost = torch.nn.functional.cross_entropy(logits, labels)

    return cost


Y = torch.zeros((6, 1))

assert (X.shape[1] == Y.shape[1])
parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters)
print("Z3 shape =" + str(Z3.shape))
cost = compute_cost(Z3, Y)
print("cost = " + str(cost))


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=800, minibatch_size=32, print_cost=True):

    # (n_x: input size, m : number of examples in the train set)
    torch.manual_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    X = torch.from_numpy(X_train)
    Y = torch.from_numpy(Y_train)

    parameters = initialize_parameters()

    optimizer = torch.optim.Adam(params=list(parameters.values()), lr=learning_rate)

    for epoch in range(num_epochs):
        
        epoch_cost = 0.                       # Defines a cost related to an epoch
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, minibatch_size, seed)
        
        for minibatch in minibatches:
            
            (minibatch_X, minibatch_Y) = minibatch
            
            Z3 = forward_propagation(minibatch_X, parameters)

            cost = compute_cost(Z3, minibatch_Y)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            epoch_cost += cost.item() / num_minibatches

        # Print the cost every epoch
        if print_cost == True and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    X = torch.from_numpy(X_train)
    Y = torch.from_numpy(Y_train)

    print("X_train shape:", X.shape)
    print("Y_train shape:", Y.shape)
    print("Y_train type:", type(Y))

    with torch.no_grad():
        Z3 = forward_propagation(X_train, parameters)  # 重新前向传播
        predictions = torch.argmax(Z3, dim=0)
        Y_train_labels = torch.argmax(Y, dim=0)
        train_acc = (predictions == Y_train_labels).float().mean()
        print("X_train shape:", predictions.shape)
        print("Y_train shape:", Y_train_labels.shape)
        print ("Train Accuracy:", str(train_acc))
    
    
    X = torch.from_numpy(X_test)
    Y = torch.from_numpy(Y_test)
    
    print("X_train shape:", X.shape)
    print("Y_train shape:", Y.shape)
    print("Y_train type:", type(Y))
    with torch.no_grad():
        Z3 = forward_propagation(X_test, parameters)  # 重新前向传播
        predictions = torch.argmax(Z3, dim=0)
        Y_test_labels = torch.argmax(Y, dim=0)
        test_acc = (predictions == Y_test_labels).float().mean()
        print ("Test Accuracy:", str(test_acc))
    
    return parameters

torch.set_default_dtype(torch.float64)
parameters = model(X_train, Y_train, X_test, Y_test)
                
                