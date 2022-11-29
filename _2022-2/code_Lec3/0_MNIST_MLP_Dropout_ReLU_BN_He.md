---
layout: default
title:  "0_MNIST_MLP_Dropout_ReLU_BN_He"
date: 2022-11-22 01:11:34 +0900
collection: 2022-2
usemathjax: true
last_modified_date: 2022-11-14 +0900
nav_order: 99
parent: 2022 Fall Seminar
author: 권기혁
# nav_exclude: true
---
```python
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
```


```python
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
```

    Using PyTorch version: 1.12.1+cu113  Device: cuda
    


```python
BATCH_SIZE = 32
EPOCHS = 20
```


```python
train_dataset = datasets.MNIST(root = "../data/MNIST",
                               train = True,
                               download = True,
                               transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = "../data/MNIST",
                              train = False,
                              transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
    


      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ../data/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
    


      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ../data/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
    


      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ../data/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
    


      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ../data/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/MNIST/raw
    
    


```python
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break
```

    X_train: torch.Size([32, 1, 28, 28]) type: torch.FloatTensor
    y_train: torch.Size([32]) type: torch.LongTensor
    


```python
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap = "gray_r")
    plt.title('Class: ' + str(y_train[i].item()))
```


    
![png](output_5_0.png)
    



```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x
```


```python
import torch.nn.init as init
def weight_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)

model = Net().to(DEVICE)
model.apply(weight_init)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
criterion = nn.CrossEntropyLoss()

print(model)
```

    Net(
      (fc1): Linear(in_features=784, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=256, bias=True)
      (fc3): Linear(in_features=256, out_features=10, bias=True)
      (batch_norm1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (batch_norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    


```python
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image), 
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item()))
```


```python
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= (len(test_loader.dataset) / BATCH_SIZE)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy
```


```python
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))
```

    Train Epoch: 1 [0/60000 (0%)]	Train Loss: 2.982483
    Train Epoch: 1 [6400/60000 (11%)]	Train Loss: 0.570645
    Train Epoch: 1 [12800/60000 (21%)]	Train Loss: 0.502169
    Train Epoch: 1 [19200/60000 (32%)]	Train Loss: 0.607567
    Train Epoch: 1 [25600/60000 (43%)]	Train Loss: 0.842404
    Train Epoch: 1 [32000/60000 (53%)]	Train Loss: 0.649236
    Train Epoch: 1 [38400/60000 (64%)]	Train Loss: 0.427547
    Train Epoch: 1 [44800/60000 (75%)]	Train Loss: 0.403249
    Train Epoch: 1 [51200/60000 (85%)]	Train Loss: 0.766969
    Train Epoch: 1 [57600/60000 (96%)]	Train Loss: 0.612467
    
    [EPOCH: 1], 	Test Loss: 0.2224, 	Test Accuracy: 93.43 % 
    
    Train Epoch: 2 [0/60000 (0%)]	Train Loss: 0.538040
    Train Epoch: 2 [6400/60000 (11%)]	Train Loss: 0.444967
    Train Epoch: 2 [12800/60000 (21%)]	Train Loss: 0.231676
    Train Epoch: 2 [19200/60000 (32%)]	Train Loss: 0.297091
    Train Epoch: 2 [25600/60000 (43%)]	Train Loss: 0.449919
    Train Epoch: 2 [32000/60000 (53%)]	Train Loss: 0.678980
    Train Epoch: 2 [38400/60000 (64%)]	Train Loss: 0.345246
    Train Epoch: 2 [44800/60000 (75%)]	Train Loss: 0.660861
    Train Epoch: 2 [51200/60000 (85%)]	Train Loss: 0.230832
    Train Epoch: 2 [57600/60000 (96%)]	Train Loss: 0.255535
    
    [EPOCH: 2], 	Test Loss: 0.1728, 	Test Accuracy: 94.67 % 
    
    Train Epoch: 3 [0/60000 (0%)]	Train Loss: 0.326840
    Train Epoch: 3 [6400/60000 (11%)]	Train Loss: 0.136414
    Train Epoch: 3 [12800/60000 (21%)]	Train Loss: 0.211074
    Train Epoch: 3 [19200/60000 (32%)]	Train Loss: 0.141939
    Train Epoch: 3 [25600/60000 (43%)]	Train Loss: 0.561737
    Train Epoch: 3 [32000/60000 (53%)]	Train Loss: 0.338441
    Train Epoch: 3 [38400/60000 (64%)]	Train Loss: 0.323387
    Train Epoch: 3 [44800/60000 (75%)]	Train Loss: 0.347304
    Train Epoch: 3 [51200/60000 (85%)]	Train Loss: 0.084859
    Train Epoch: 3 [57600/60000 (96%)]	Train Loss: 0.089178
    
    [EPOCH: 3], 	Test Loss: 0.1489, 	Test Accuracy: 95.45 % 
    
    Train Epoch: 4 [0/60000 (0%)]	Train Loss: 0.065480
    Train Epoch: 4 [6400/60000 (11%)]	Train Loss: 0.334499
    Train Epoch: 4 [12800/60000 (21%)]	Train Loss: 0.386923
    Train Epoch: 4 [19200/60000 (32%)]	Train Loss: 0.739575
    Train Epoch: 4 [25600/60000 (43%)]	Train Loss: 0.550873
    Train Epoch: 4 [32000/60000 (53%)]	Train Loss: 0.169128
    Train Epoch: 4 [38400/60000 (64%)]	Train Loss: 0.401474
    Train Epoch: 4 [44800/60000 (75%)]	Train Loss: 0.469707
    Train Epoch: 4 [51200/60000 (85%)]	Train Loss: 0.184209
    Train Epoch: 4 [57600/60000 (96%)]	Train Loss: 0.106019
    
    [EPOCH: 4], 	Test Loss: 0.1283, 	Test Accuracy: 96.00 % 
    
    Train Epoch: 5 [0/60000 (0%)]	Train Loss: 0.289607
    Train Epoch: 5 [6400/60000 (11%)]	Train Loss: 0.281095
    Train Epoch: 5 [12800/60000 (21%)]	Train Loss: 0.215075
    Train Epoch: 5 [19200/60000 (32%)]	Train Loss: 0.356820
    Train Epoch: 5 [25600/60000 (43%)]	Train Loss: 0.160610
    Train Epoch: 5 [32000/60000 (53%)]	Train Loss: 0.316136
    Train Epoch: 5 [38400/60000 (64%)]	Train Loss: 0.165865
    Train Epoch: 5 [44800/60000 (75%)]	Train Loss: 0.226877
    Train Epoch: 5 [51200/60000 (85%)]	Train Loss: 0.207517
    Train Epoch: 5 [57600/60000 (96%)]	Train Loss: 0.154142
    
    [EPOCH: 5], 	Test Loss: 0.1162, 	Test Accuracy: 96.37 % 
    
    Train Epoch: 6 [0/60000 (0%)]	Train Loss: 0.326900
    Train Epoch: 6 [6400/60000 (11%)]	Train Loss: 0.095518
    Train Epoch: 6 [12800/60000 (21%)]	Train Loss: 0.073130
    Train Epoch: 6 [19200/60000 (32%)]	Train Loss: 0.311288
    Train Epoch: 6 [25600/60000 (43%)]	Train Loss: 0.196510
    Train Epoch: 6 [32000/60000 (53%)]	Train Loss: 0.118644
    Train Epoch: 6 [38400/60000 (64%)]	Train Loss: 0.224573
    Train Epoch: 6 [44800/60000 (75%)]	Train Loss: 0.183870
    Train Epoch: 6 [51200/60000 (85%)]	Train Loss: 0.087863
    Train Epoch: 6 [57600/60000 (96%)]	Train Loss: 0.357086
    
    [EPOCH: 6], 	Test Loss: 0.1078, 	Test Accuracy: 96.73 % 
    
    Train Epoch: 7 [0/60000 (0%)]	Train Loss: 0.198536
    Train Epoch: 7 [6400/60000 (11%)]	Train Loss: 0.090745
    Train Epoch: 7 [12800/60000 (21%)]	Train Loss: 0.262080
    Train Epoch: 7 [19200/60000 (32%)]	Train Loss: 0.068609
    Train Epoch: 7 [25600/60000 (43%)]	Train Loss: 0.590940
    Train Epoch: 7 [32000/60000 (53%)]	Train Loss: 0.426208
    Train Epoch: 7 [38400/60000 (64%)]	Train Loss: 0.041000
    Train Epoch: 7 [44800/60000 (75%)]	Train Loss: 0.291964
    Train Epoch: 7 [51200/60000 (85%)]	Train Loss: 0.072686
    Train Epoch: 7 [57600/60000 (96%)]	Train Loss: 0.023258
    
    [EPOCH: 7], 	Test Loss: 0.1049, 	Test Accuracy: 96.84 % 
    
    Train Epoch: 8 [0/60000 (0%)]	Train Loss: 0.154875
    Train Epoch: 8 [6400/60000 (11%)]	Train Loss: 0.374769
    Train Epoch: 8 [12800/60000 (21%)]	Train Loss: 0.220888
    Train Epoch: 8 [19200/60000 (32%)]	Train Loss: 0.307588
    Train Epoch: 8 [25600/60000 (43%)]	Train Loss: 0.154757
    Train Epoch: 8 [32000/60000 (53%)]	Train Loss: 0.190583
    Train Epoch: 8 [38400/60000 (64%)]	Train Loss: 0.062868
    Train Epoch: 8 [44800/60000 (75%)]	Train Loss: 0.107506
    Train Epoch: 8 [51200/60000 (85%)]	Train Loss: 0.128710
    Train Epoch: 8 [57600/60000 (96%)]	Train Loss: 0.114214
    
    [EPOCH: 8], 	Test Loss: 0.0992, 	Test Accuracy: 96.99 % 
    
    Train Epoch: 9 [0/60000 (0%)]	Train Loss: 0.306962
    Train Epoch: 9 [6400/60000 (11%)]	Train Loss: 0.167397
    Train Epoch: 9 [12800/60000 (21%)]	Train Loss: 0.285983
    Train Epoch: 9 [19200/60000 (32%)]	Train Loss: 0.096924
    Train Epoch: 9 [25600/60000 (43%)]	Train Loss: 0.078761
    Train Epoch: 9 [32000/60000 (53%)]	Train Loss: 0.148236
    Train Epoch: 9 [38400/60000 (64%)]	Train Loss: 0.045097
    Train Epoch: 9 [44800/60000 (75%)]	Train Loss: 0.197048
    Train Epoch: 9 [51200/60000 (85%)]	Train Loss: 0.331972
    Train Epoch: 9 [57600/60000 (96%)]	Train Loss: 0.049501
    
    [EPOCH: 9], 	Test Loss: 0.0902, 	Test Accuracy: 97.20 % 
    
    Train Epoch: 10 [0/60000 (0%)]	Train Loss: 0.182756
    Train Epoch: 10 [6400/60000 (11%)]	Train Loss: 0.034406
    Train Epoch: 10 [12800/60000 (21%)]	Train Loss: 0.130722
    Train Epoch: 10 [19200/60000 (32%)]	Train Loss: 0.418017
    Train Epoch: 10 [25600/60000 (43%)]	Train Loss: 0.103932
    Train Epoch: 10 [32000/60000 (53%)]	Train Loss: 0.092033
    Train Epoch: 10 [38400/60000 (64%)]	Train Loss: 0.143645
    Train Epoch: 10 [44800/60000 (75%)]	Train Loss: 0.149150
    Train Epoch: 10 [51200/60000 (85%)]	Train Loss: 0.532430
    Train Epoch: 10 [57600/60000 (96%)]	Train Loss: 0.083848
    
    [EPOCH: 10], 	Test Loss: 0.0857, 	Test Accuracy: 97.33 % 
    
    Train Epoch: 11 [0/60000 (0%)]	Train Loss: 0.224638
    Train Epoch: 11 [6400/60000 (11%)]	Train Loss: 0.061418
    Train Epoch: 11 [12800/60000 (21%)]	Train Loss: 0.135800
    Train Epoch: 11 [19200/60000 (32%)]	Train Loss: 0.104722
    Train Epoch: 11 [25600/60000 (43%)]	Train Loss: 0.030334
    Train Epoch: 11 [32000/60000 (53%)]	Train Loss: 0.114534
    Train Epoch: 11 [38400/60000 (64%)]	Train Loss: 0.067933
    Train Epoch: 11 [44800/60000 (75%)]	Train Loss: 0.062294
    Train Epoch: 11 [51200/60000 (85%)]	Train Loss: 0.164575
    Train Epoch: 11 [57600/60000 (96%)]	Train Loss: 0.128098
    
    [EPOCH: 11], 	Test Loss: 0.0837, 	Test Accuracy: 97.36 % 
    
    Train Epoch: 12 [0/60000 (0%)]	Train Loss: 0.099557
    Train Epoch: 12 [6400/60000 (11%)]	Train Loss: 0.063159
    Train Epoch: 12 [12800/60000 (21%)]	Train Loss: 0.053264
    Train Epoch: 12 [19200/60000 (32%)]	Train Loss: 0.073025
    Train Epoch: 12 [25600/60000 (43%)]	Train Loss: 0.091995
    Train Epoch: 12 [32000/60000 (53%)]	Train Loss: 0.076850
    Train Epoch: 12 [38400/60000 (64%)]	Train Loss: 0.108102
    Train Epoch: 12 [44800/60000 (75%)]	Train Loss: 0.085792
    Train Epoch: 12 [51200/60000 (85%)]	Train Loss: 0.088433
    Train Epoch: 12 [57600/60000 (96%)]	Train Loss: 0.147557
    
    [EPOCH: 12], 	Test Loss: 0.0806, 	Test Accuracy: 97.51 % 
    
    Train Epoch: 13 [0/60000 (0%)]	Train Loss: 0.032560
    Train Epoch: 13 [6400/60000 (11%)]	Train Loss: 0.142005
    Train Epoch: 13 [12800/60000 (21%)]	Train Loss: 0.063277
    Train Epoch: 13 [19200/60000 (32%)]	Train Loss: 0.178761
    Train Epoch: 13 [25600/60000 (43%)]	Train Loss: 0.078791
    Train Epoch: 13 [32000/60000 (53%)]	Train Loss: 0.066515
    Train Epoch: 13 [38400/60000 (64%)]	Train Loss: 0.250516
    Train Epoch: 13 [44800/60000 (75%)]	Train Loss: 0.292390
    Train Epoch: 13 [51200/60000 (85%)]	Train Loss: 0.130400
    Train Epoch: 13 [57600/60000 (96%)]	Train Loss: 0.276094
    
    [EPOCH: 13], 	Test Loss: 0.0766, 	Test Accuracy: 97.64 % 
    
    Train Epoch: 14 [0/60000 (0%)]	Train Loss: 0.202264
    Train Epoch: 14 [6400/60000 (11%)]	Train Loss: 0.119045
    Train Epoch: 14 [12800/60000 (21%)]	Train Loss: 0.158097
    Train Epoch: 14 [19200/60000 (32%)]	Train Loss: 0.182480
    Train Epoch: 14 [25600/60000 (43%)]	Train Loss: 0.104634
    Train Epoch: 14 [32000/60000 (53%)]	Train Loss: 0.060589
    Train Epoch: 14 [38400/60000 (64%)]	Train Loss: 0.259142
    Train Epoch: 14 [44800/60000 (75%)]	Train Loss: 0.035314
    Train Epoch: 14 [51200/60000 (85%)]	Train Loss: 0.091466
    Train Epoch: 14 [57600/60000 (96%)]	Train Loss: 0.069200
    
    [EPOCH: 14], 	Test Loss: 0.0771, 	Test Accuracy: 97.60 % 
    
    Train Epoch: 15 [0/60000 (0%)]	Train Loss: 0.036656
    Train Epoch: 15 [6400/60000 (11%)]	Train Loss: 0.172608
    Train Epoch: 15 [12800/60000 (21%)]	Train Loss: 0.494084
    Train Epoch: 15 [19200/60000 (32%)]	Train Loss: 0.108016
    Train Epoch: 15 [25600/60000 (43%)]	Train Loss: 0.030475
    Train Epoch: 15 [32000/60000 (53%)]	Train Loss: 0.050303
    Train Epoch: 15 [38400/60000 (64%)]	Train Loss: 0.089965
    Train Epoch: 15 [44800/60000 (75%)]	Train Loss: 0.160547
    Train Epoch: 15 [51200/60000 (85%)]	Train Loss: 0.138789
    Train Epoch: 15 [57600/60000 (96%)]	Train Loss: 0.133560
    
    [EPOCH: 15], 	Test Loss: 0.0730, 	Test Accuracy: 97.77 % 
    
    Train Epoch: 16 [0/60000 (0%)]	Train Loss: 0.107885
    Train Epoch: 16 [6400/60000 (11%)]	Train Loss: 0.191566
    Train Epoch: 16 [12800/60000 (21%)]	Train Loss: 0.086493
    Train Epoch: 16 [19200/60000 (32%)]	Train Loss: 0.248595
    Train Epoch: 16 [25600/60000 (43%)]	Train Loss: 0.147690
    Train Epoch: 16 [32000/60000 (53%)]	Train Loss: 0.199059
    Train Epoch: 16 [38400/60000 (64%)]	Train Loss: 0.257633
    Train Epoch: 16 [44800/60000 (75%)]	Train Loss: 0.201671
    Train Epoch: 16 [51200/60000 (85%)]	Train Loss: 0.185560
    Train Epoch: 16 [57600/60000 (96%)]	Train Loss: 0.309864
    
    [EPOCH: 16], 	Test Loss: 0.0725, 	Test Accuracy: 97.76 % 
    
    Train Epoch: 17 [0/60000 (0%)]	Train Loss: 0.197708
    Train Epoch: 17 [6400/60000 (11%)]	Train Loss: 0.060845
    Train Epoch: 17 [12800/60000 (21%)]	Train Loss: 0.047888
    Train Epoch: 17 [19200/60000 (32%)]	Train Loss: 0.194344
    Train Epoch: 17 [25600/60000 (43%)]	Train Loss: 0.109386
    Train Epoch: 17 [32000/60000 (53%)]	Train Loss: 0.130850
    Train Epoch: 17 [38400/60000 (64%)]	Train Loss: 0.093416
    Train Epoch: 17 [44800/60000 (75%)]	Train Loss: 0.121703
    Train Epoch: 17 [51200/60000 (85%)]	Train Loss: 0.070911
    Train Epoch: 17 [57600/60000 (96%)]	Train Loss: 0.043108
    
    [EPOCH: 17], 	Test Loss: 0.0695, 	Test Accuracy: 97.94 % 
    
    Train Epoch: 18 [0/60000 (0%)]	Train Loss: 0.233003
    Train Epoch: 18 [6400/60000 (11%)]	Train Loss: 0.081782
    Train Epoch: 18 [12800/60000 (21%)]	Train Loss: 0.038036
    Train Epoch: 18 [19200/60000 (32%)]	Train Loss: 0.029246
    Train Epoch: 18 [25600/60000 (43%)]	Train Loss: 0.149569
    Train Epoch: 18 [32000/60000 (53%)]	Train Loss: 0.014726
    Train Epoch: 18 [38400/60000 (64%)]	Train Loss: 0.115903
    Train Epoch: 18 [44800/60000 (75%)]	Train Loss: 0.093841
    Train Epoch: 18 [51200/60000 (85%)]	Train Loss: 0.099389
    Train Epoch: 18 [57600/60000 (96%)]	Train Loss: 0.311888
    
    [EPOCH: 18], 	Test Loss: 0.0684, 	Test Accuracy: 97.91 % 
    
    Train Epoch: 19 [0/60000 (0%)]	Train Loss: 0.046956
    Train Epoch: 19 [6400/60000 (11%)]	Train Loss: 0.098525
    Train Epoch: 19 [12800/60000 (21%)]	Train Loss: 0.050284
    Train Epoch: 19 [19200/60000 (32%)]	Train Loss: 0.325607
    Train Epoch: 19 [25600/60000 (43%)]	Train Loss: 0.062045
    Train Epoch: 19 [32000/60000 (53%)]	Train Loss: 0.098317
    Train Epoch: 19 [38400/60000 (64%)]	Train Loss: 0.071337
    Train Epoch: 19 [44800/60000 (75%)]	Train Loss: 0.097055
    Train Epoch: 19 [51200/60000 (85%)]	Train Loss: 0.022128
    Train Epoch: 19 [57600/60000 (96%)]	Train Loss: 0.243198
    
    [EPOCH: 19], 	Test Loss: 0.0657, 	Test Accuracy: 98.00 % 
    
    Train Epoch: 20 [0/60000 (0%)]	Train Loss: 0.281539
    Train Epoch: 20 [6400/60000 (11%)]	Train Loss: 0.085662
    Train Epoch: 20 [12800/60000 (21%)]	Train Loss: 0.117857
    Train Epoch: 20 [19200/60000 (32%)]	Train Loss: 0.037888
    Train Epoch: 20 [25600/60000 (43%)]	Train Loss: 0.207817
    Train Epoch: 20 [32000/60000 (53%)]	Train Loss: 0.082747
    Train Epoch: 20 [38400/60000 (64%)]	Train Loss: 0.106170
    Train Epoch: 20 [44800/60000 (75%)]	Train Loss: 0.077973
    Train Epoch: 20 [51200/60000 (85%)]	Train Loss: 0.174070
    Train Epoch: 20 [57600/60000 (96%)]	Train Loss: 0.273948
    
    [EPOCH: 20], 	Test Loss: 0.0660, 	Test Accuracy: 97.99 % 
    
    


```python

```
