# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import pandas as pd
import numpy as np

# train data (XOR Problem)
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# input - hidden Layer
w1 = np.random.randn(2,2)
b1 = np.random.randn(2,2)

# hidden - ouput layer
w2 = np.random.randn(1,2)
b2 = np.random.randn(1)

# Epoch
ep = 20000

# Learning Rate
lr = 1
mse = []

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range(ep):
        E = np.array([])
        result = np.array([])

        for j in range(len(x)):
            Ha = np.array([])

            # Feed Foward
            # Input - Hidden Layer
            for k in range(len(w1)):
                Ha = np.append(Ha, 1 / (1 + np.exp(-(np.sum(x[j] * w1[k]) + b1[0][k]))))
                # Use Sigmoid Function

            # Hidden - Output Layer
            Hb = 1 / (1 + np.exp(-(np.sum(Ha * w2) + b2)))

            # Error
            E = np.append(E , y[j] - Hb)
            result = np.append(result, Hb)