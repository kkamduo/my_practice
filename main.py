# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import pandas as pd
import numpy as np

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
    a = np.array(range(0, 6))

    print(a)
