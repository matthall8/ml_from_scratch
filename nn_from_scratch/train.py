#%%
import numpy as np # linear algebra
from dataset import get_data
from model import MLP

# Hyper parameters
iterations = 200
learning_rate = 0.10
hidden_layer_size = 50

# Get Training Data
Y_train, X_train, m_train = get_data()

# Helper functions for testing model accuracy

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Gradient descent training looop

def gradient_descent(X, Y, alpha, iterations):
    input_params = X.shape[0]
    model = MLP(m_train, hidden_layer_size,input_params)
    W1, b1, W2, b2 = model.init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = model.forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = model.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = model.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Training Accuracy:", get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, learning_rate, iterations)