import numpy as np
import pandas as pd

data = pd.read_csv('~/downloads/PythonPractice/FootballNeuralNetwork/2022-2023 Football Player Stats.csv', encoding='latin-1', sep=';')
#data.head()
data = data[['Pos', 'Goals', 'Shots', 'SoT', 'SoT%', 'G/Sh', 'G/SoT', 'PasTotCmp', 'PasTotAtt', 'PasTotCmp%', 'Assists', 'PasAss', 'Tkl', 'TklWon', 'Blocks', 'BlkSh', 'BlkPass', 'Int', 'Tkl+Int', 'Clr', 'Touches', 'TouDef3rd', 'TouMid3rd', 'TouAtt3rd', 'TouAttPen', 'Carries', 'Rec', 'TklW', 'Recov', 'AerWon']]
data['Pos'] = data['Pos'].apply(lambda x: x[:2])

data = np.array(data)
m, n = data.shape
print(m)
print(n)
#m =  number of rows
#n = number of columns + 1
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


"""
def init_params():
    W1 = np.random.rand(4, 29) - 0.5
    b1 = np.random.rand(4, 1) - 0.5
    W2 = np.random.rand(4, 4) - 0.5
    b2 = np.random.rand(4, 1) - 0.5
    return W1, b1, W2, b2
"""
def init_params():
    W1 = np.random.rand(14, 29) - 0.5
    b1 = np.random.rand(14, 1) - 0.5
    W2 = np.random.rand(7, 14) - 0.5
    b2 = np.random.rand(7, 1) - 0.5
    W3 = np.random.rand(4, 7) - 0.5
    b3 = np.random.rand(4, 1) - 0.5
    return W1, b1, W2, b2, W3, b3


def ReLU(Z):
    return np.maximum(0, Z)

#def ReLU(Z):
#    Z = Z.astype(float)
#    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    Z = Z.astype(float)
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # for numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


"""
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
"""
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def ReLU_deriv(Z):
    return Z > 0
#def ReLU_deriv(Z):
#    fz = ReLU(Z)
#    return fz * (1 - fz)

def create_mapping(Y):
    unique_strings = np.unique(Y)
    string_to_index = {string: index for index, string in enumerate(unique_strings)}
    return string_to_index

def convert_to_indices(Y, string_to_index):
    return np.array([string_to_index[string] for string in Y])

def one_hot(Y):
    uniqueY = create_mapping(Y)
    Y = convert_to_indices(Y, uniqueY)
    one_hot_Y = np.zeros((Y.size, 4))
    """if Y == 'FW':
        Y = 0
    elif Y == 'MF':
        Y = 1
    elif Y == 'DF':
        Y = 2
    else:
        Y = 3
    """
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
"""
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y 
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
"""
def back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y 
    dW3 = 1/m * dZ3.dot(A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)
    
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3
"""
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1, b1, W2, b2
"""
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

"""
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: " + str(i))
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, convert_to_indices(Y, create_mapping(Y)))
            print(f"Accuracy: {accuracy}")

    return W1, b1, W2, b2
"""
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2, W3, b3 = init_params()
    Y_indices = convert_to_indices(Y, create_mapping(Y))  # Convert Y to indices once
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 50 == 0:
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y_indices)
            print(f"Iteration: {i}")
            print(f"Accuracy: {accuracy}")
    return W1, b1, W2, b2, W3, b3

#W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.01)
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 1000, 0.01)  # Adjusted learning rate
