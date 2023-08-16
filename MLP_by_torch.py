import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device is ",device)

random.seed(52)
torch.manual_seed(52)

if device == 'cuda':
    torch.cuda.manual_seed_all(52)

lrs = 0.0001
epochs = 15
batch_size = 100
drop_prob = 0.3

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train = True,
                          transform=transforms.ToTensor(),
                          download=True
                          )

mnist_test = dsets.MNIST(root='MNIST_data/',
                          train = False,
                          transform=transforms.ToTensor(),
                          download=True
                          )

data_loader = DataLoader(dataset=mnist_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                        )

lin1 = nn.Linear(784,512, bias=True)
lin2 = nn.Linear(512,512, bias=True)
lin3 = nn.Linear(512,512, bias=True)
lin4 = nn.Linear(512,10, bias=True)

relu = nn.ReLU()
dropout = nn.Dropout(p=drop_prob)

nn.init.xavier_uniform_(lin1.weight)
nn.init.xavier_uniform_(lin2.weight)
nn.init.xavier_uniform_(lin3.weight)
nn.init.xavier_uniform_(lin4.weight)

model = nn.Sequential(lin1,relu,dropout,
                      lin2,relu,dropout,
                      lin3,relu,dropout,
                      lin4
                      )

model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lrs)

total_batch = len(data_loader)
model.train()

for epoch in range(epochs):
    avg_cost = 0

    for X,Y in data_loader:
        X = X.view(-1,28*28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis,Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch :','%04d' % (epoch+1), 'cost = ','{:.9f}'.format(avg_cost))

model_script = torch.jit.script(model)
model_script.save("/root/model/test1.pt")

print('Save the Model')
print('Done')