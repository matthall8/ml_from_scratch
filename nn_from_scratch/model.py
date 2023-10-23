#%%
import numpy as np # linear algebra

class MLP():
    def __init__(self, m_train, hidden_layer_size,input_params):
        self.m = m_train
        self.hidden_layer_size = hidden_layer_size
        self.input_params = input_params

    def init_params(self):
        W1 = np.random.randn(self.hidden_layer_size, self.input_params)
        b1 = np.random.randn(self.hidden_layer_size, 1)
        W2 = np.random.randn(10, self.hidden_layer_size)
        b2 = np.random.randn(10, 1)
        return W1, b1, W2, b2

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
        
    def forward_prop(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def ReLU_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / self.m * dZ2.dot(A1.T)
        db2 = 1 / self.m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / self.m * dZ1.dot(X.T)
        db1 = 1 / self.m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1    
        W2 = W2 - alpha * dW2  
        b2 = b2 - alpha * db2    
        return W1, b1, W2, b2