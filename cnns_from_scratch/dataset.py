#%%
from keras.datasets import mnist

#%%
def get_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    train_images = ((X_train / 255) - 0.5)
    train_labels = Y_train
    test_images = ((X_test / 255) - 0.5)
    test_labels = Y_test
    return train_images, train_labels, test_images, test_labels