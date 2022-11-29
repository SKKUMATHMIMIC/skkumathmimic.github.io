---
layout: default
title:  "6_softmax_classification"
date: 2022-11-22 01:11:34 +0900
collection: 2022-2
usemathjax: true
last_modified_date: 2022-11-14 +0900
nav_order: 96
parent: 2022 Fall Seminar
author: 권기혁
# nav_exclude: true
---
# 6.Softmax Classification

## Imports


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```


```python
# For reproducibility
torch.manual_seed(1)
```




    <torch._C.Generator at 0x7faba8602ad0>



## Softmax

Convert numbers to probabilities with softmax.

$$ P(class=i) = \frac{e^i}{\sum e^i} $$


```python
z = torch.FloatTensor([1, 2, 3])
```

PyTorch has a `softmax` function.


```python
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
```

    tensor([0.0900, 0.2447, 0.6652])
    

Since they are probabilities, they should add up to 1. Let's do a sanity check.


```python
hypothesis.sum()
```




    tensor(1.)



## Cross Entropy Loss (Low-level)

For multi-class classification, we use the cross entropy loss.

$$ L = \frac{1}{N} \sum - y \log(\hat{y}) $$

where $\hat{y}$ is the predicted probability and $y$ is the correct probability (0 or 1).


```python
z = torch.rand(3, 5, requires_grad=True)
print(z)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
```

    tensor([[0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
            [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
            [0.6387, 0.5247, 0.6826, 0.3051, 0.4635]], requires_grad=True)
    tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
            [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
            [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward0>)
    


```python
y = torch.randint(5, (3,)).long()
print(y)
```

    tensor([0, 2, 1])
    


```python
y_one_hot = torch.zeros_like(hypothesis)
print(y.shape)
print(y)
print(y.unsqueeze(1).shape)
print(y.unsqueeze(1))
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
```

    torch.Size([3])
    tensor([0, 2, 1])
    torch.Size([3, 1])
    tensor([[0],
            [2],
            [1]])
    




    tensor([[1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0.]])




```python
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
```

    tensor(1.4689, grad_fn=<MeanBackward0>)
    

## Cross-entropy Loss with `torch.nn.functional`

PyTorch has `F.log_softmax()` function.


```python
# Low level
torch.log(F.softmax(z, dim=1))
```




    tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
            [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
            [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]], grad_fn=<LogBackward0>)




```python
# High level
F.log_softmax(z, dim=1)
```




    tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
            [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
            [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]],
           grad_fn=<LogSoftmaxBackward0>)



PyTorch also has `F.nll_loss()` function that computes the negative loss likelihood.


```python
# Low level
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
```




    tensor(1.4689, grad_fn=<MeanBackward0>)




```python
# High level
F.nll_loss(F.log_softmax(z, dim=1), y)
```




    tensor(1.4689, grad_fn=<NllLossBackward0>)



PyTorch also has `F.cross_entropy` that combines `F.log_softmax()` and `F.nll_loss()`.


```python
F.cross_entropy(z, y)
```




    tensor(1.4689, grad_fn=<NllLossBackward0>)



## Training with Low-level Cross Entropy Loss


```python
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
```


```python
# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True) # 4개의 열 (feature) 그리고 3개의 클래스
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 5000
for epoch in range(nb_epochs + 1):

    # Cost 계산 (1)
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1) # or .mm or @
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

    Epoch    0/5000 Cost: 1.098612
    Epoch  500/5000 Cost: 0.774822
    Epoch 1000/5000 Cost: 0.738749
    Epoch 1500/5000 Cost: 0.721463
    Epoch 2000/5000 Cost: 0.710759
    Epoch 2500/5000 Cost: 0.703385
    Epoch 3000/5000 Cost: 0.697990
    Epoch 3500/5000 Cost: 0.693876
    Epoch 4000/5000 Cost: 0.690640
    Epoch 4500/5000 Cost: 0.688032
    Epoch 5000/5000 Cost: 0.685888
    

## Training with `F.cross_entropy`


```python
# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 5000
for epoch in range(nb_epochs + 1):

    # Cost 계산 (2)
    z = x_train.matmul(W) + b # or .mm or @
    cost = F.cross_entropy(z, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

    Epoch    0/5000 Cost: 1.098612
    Epoch  500/5000 Cost: 0.568255
    Epoch 1000/5000 Cost: 0.399962
    Epoch 1500/5000 Cost: 0.285279
    Epoch 2000/5000 Cost: 0.246098
    Epoch 2500/5000 Cost: 0.216146
    Epoch 3000/5000 Cost: 0.192427
    Epoch 3500/5000 Cost: 0.173168
    Epoch 4000/5000 Cost: 0.157232
    Epoch 4500/5000 Cost: 0.143840
    Epoch 5000/5000 Cost: 0.132443
    

## High-level Implementation with `nn.Module`


```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output dim = 3

    def forward(self, x):
        return self.linear(x)
```


```python
model = SoftmaxClassifierModel()
```


```python
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 5000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

    Epoch    0/5000 Cost: 1.849513
    Epoch  500/5000 Cost: 0.451947
    Epoch 1000/5000 Cost: 0.241922
    Epoch 1500/5000 Cost: 0.190806
    Epoch 2000/5000 Cost: 0.157194
    Epoch 2500/5000 Cost: 0.133385
    Epoch 3000/5000 Cost: 0.115685
    Epoch 3500/5000 Cost: 0.102039
    Epoch 4000/5000 Cost: 0.091213
    Epoch 4500/5000 Cost: 0.082423
    Epoch 5000/5000 Cost: 0.075151
    


```python

```
