import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

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

# Class my_cnn will inheritage the nn.module that in torch.nn
class my_cnn(nn.Module):
    def __init__ (self):
        # nn.Module are my_cnn's parent.
        # Convolution layers are play for find features fo input images.
        # FC layers play a role in arranging the feature maps before classification by hidden layers, similar to a DNN.

        super(my_cnn,self).__init__()
        self.conv1 = nn.Conv2d(1,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(4 * 4 * 50, 200)
        self.fc2 = nn.Linear(200,100)

    def forward(self,x):
        # You can create as many convolutional layers as you want.

        # Max pooling explain. (Oct 5)

        # STEP 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)

        # STEP 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)

        # STEP 3
        x = x.view(-1, 4 * 4 * 50) # [Batch_size, 50, 4, 4]

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
    
if __name__ == "__main__":
    print(torch.cuda.is_available())

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Train data
    train_data = datasets.MNIST('./data/', train = True, download = True, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ]))
    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = 50, shuffle = True)

    # Test data
    test_data = datasets.MNIST('./data/', train = False, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ]))
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 50, shuffle = True)

    cnn = my_cnn()
    # Define the Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr = 0.01)

    cnn.train()

    for epoch in range(5):
        for idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = cnn(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print("loss of {} epoch, {} idx : {}".format(epoch+1,idx,loss.item()))

    cnn.eval()  # Protect to train test case
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = cnn(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))