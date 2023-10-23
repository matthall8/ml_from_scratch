#%%
import numpy as np # linear algebra
from dataset import get_data
from model import Conv_Op, Max_Pool, Softmax

# Hyper parameters
epochs = 5
learning_rate = 0.05

# Get Training Data
train_images, train_labels, test_images, test_labels = get_data()

# Create Model instances

conv = Conv_Op(8,3)
max_pool = Max_Pool(2)
softmax = Softmax(13*13*8, 10)

# Gradient descent training looop

def cnn_forward_prop(image, label):
    out_p = conv.forward_prop(image)
    out_p = max_pool.forward_prop(out_p)
    out_p = softmax.forward_prop(out_p)

    # Calculate cross entropy loss

    cross_ent_loss = -np.log(out_p[label])
    accuracy_eval = 1 if np.argmax(out_p) == label else 0

    return out_p, cross_ent_loss, accuracy_eval

def training_cnn(image, label, learning_rate=learning_rate):
    #forward 

    out, loss, accuracy = cnn_forward_prop(image, label)

    #calculate initial gradient

    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    #backprop 

    grad_back = softmax.back_prop(gradient, learning_rate)
    grad_back = max_pool.back_prop(grad_back)
    grad_back = conv.back_prop(grad_back, learning_rate)

    return loss, accuracy

for epoch1 in range(epochs):
    print("Epoch %d --->" % (epoch1 + 1))

    # Shuffule the data

    shuffle_data = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_data]
    train_labels = train_labels[shuffle_data]

    # Training the CNN

    loss = 0 
    num_correct = 0
    for i, (img, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 0:
            print('%d steps: Average loss: %.3f and Accuracy: %d%%' % (i+1, loss/100, num_correct))
            loss = 0
            num_correct = 0

        ll, accu = training_cnn(img, label)
        loss += ll
        num_correct += accu