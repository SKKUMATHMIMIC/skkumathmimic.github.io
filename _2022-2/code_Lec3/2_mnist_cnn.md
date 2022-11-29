---
layout: default
title:  "2_mnist_cnn"
date: 2022-11-22 01:11:34 +0900
collection: 2022-2
usemathjax: true
last_modified_date: 2022-11-14 +0900
nav_order: 101
parent: 2022 Fall Seminar
author: 권기혁
# nav_exclude: true
---
```python
# MNIST and Convolutional Neural Network
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


print('__Number CUDA Devices:', torch.cuda.device_count())
print('__CUDA Device Name:',torch.cuda.get_device_name(0))
print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
```

    cuda
    __Number CUDA Devices: 1
    __CUDA Device Name: Tesla T4
    __CUDA Device Total Memory [GB]: 15.843721216
    


```python
# parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
```


```python
# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz
    


      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz to MNIST_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz
    


      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz
    


      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    


      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw
    
    


```python
print(mnist_train)
```

    Dataset MNIST
        Number of datapoints: 60000
        Root location: MNIST_data/
        Split: Train
        StandardTransform
    Transform: ToTensor()
    


```python
# Assign batch size via dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
```


```python
# CNN Model (2 conv layers)
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        # Here ? means batch size
        self.layer1 = torch.nn.Sequential(
            #in channel = 1, out channel = 32 (# of filters), kernel size, stride, padding
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Final FC 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        # weight initialization
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc(out)
        return out
```


```python
# instantiate CNN model to the device
model = CNN().to(device)
```


```python
# define cost/loss & optimizer

criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```


```python
# print the number of batches
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))
```

    총 배치의 수 : 600
    

총 배치의 수 = 600. 그런데 배치 크기=100. 따라서 훈련 데이터는 600*100 = 60000개. 


```python
# train my model
total_batch = len(data_loader)
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('Learning Finished!')
```

    Learning started. It takes sometime.
    [Epoch:    1] cost = 0.225627884
    [Epoch:    2] cost = 0.0629666671
    [Epoch:    3] cost = 0.0462148637
    [Epoch:    4] cost = 0.0374672972
    [Epoch:    5] cost = 0.0313553065
    [Epoch:    6] cost = 0.0261589289
    [Epoch:    7] cost = 0.0217485875
    [Epoch:    8] cost = 0.0181652699
    [Epoch:    9] cost = 0.0162438396
    [Epoch:   10] cost = 0.0130265476
    [Epoch:   11] cost = 0.00996452849
    [Epoch:   12] cost = 0.00955614075
    [Epoch:   13] cost = 0.0084212441
    [Epoch:   14] cost = 0.00607894221
    [Epoch:   15] cost = 0.00688483706
    [Epoch:   16] cost = 0.00607912894
    [Epoch:   17] cost = 0.00436533429
    [Epoch:   18] cost = 0.00376819447
    [Epoch:   19] cost = 0.00484180218
    [Epoch:   20] cost = 0.00520616816
    Learning Finished!
    


```python
# Test model and check accuracy
# 학습을 진행하지 않을 것이므로 torch.no_grad()

with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
```

    Accuracy: 0.983199954032898
    

    /usr/local/lib/python3.7/dist-packages/torchvision/datasets/mnist.py:80: UserWarning: test_data has been renamed data
      warnings.warn("test_data has been renamed data")
    /usr/local/lib/python3.7/dist-packages/torchvision/datasets/mnist.py:70: UserWarning: test_labels has been renamed targets
      warnings.warn("test_labels has been renamed targets")
    


```python

```
