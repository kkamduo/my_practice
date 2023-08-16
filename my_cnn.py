import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Construction of Basic CNN are follow the under explain.

< Construction of CNN>
Input -> Conv -> Activation Function (ex : relu, mish and so on) -> Pooling ->
Conv -> Activate Function (ex : relu, mish and so on) -> Pooling -> ... ->
FC Layer -> Softmax

Convolution layers are used to extract features from the input data. 
These features are obtained by applying filters, such as edge filters, to the input data. 
These detected features are important for tasks like image classification.

Activation functions play a vital role in ML.
For example, the ReLU is commonly used function.

'''

class my_cnn(nn.Module):
    def __init__ (self):
        super(my_cnn,self).__init__()
        # Class my_cnn will inheritage the nn.module that in torch.nn
        # nn.Module are my_cnn's parent.

    def conv_layers(i_channels, out_channels, k_size, stride):
        nn.Conv2d(in_channels = i_channels, out_channels = out_channels, kernel_size = k_size, stride = stride)

    def fc_layer(x_size, y_size):
        nn.Linear(x_size, y_size)
    
    def forward(self,x):
        # You can create as many convolutional layers as you want.

        # STEP 1
        self.conv1 = self.conv(1,3,5,1)
        x = F.relu(self.conv1(x))

        # STEP 2
        self.conv2 = self.conv(3,10,5,1)
        x = F.mish(self.conv2(x))

        # STEP 3