#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

#%%
def get_data():
    data = pd.read_csv('mnist_train.csv')
    data = np.array(data)
    np.random.shuffle(data)

    data_train = data.T
    Y_train = data_train[0]
    X_train = data_train[1:]
    X_train = X_train / 255.
    _,m_train = X_train.shape
    
    return Y_train, X_train, m_train