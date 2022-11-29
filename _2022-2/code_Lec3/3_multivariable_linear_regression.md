---
layout: default
title:  "3_multivariable_linear_regression"
date: 2022-11-22 01:11:34 +0900
collection: 2022-2
usemathjax: true
last_modified_date: 2022-11-14 +0900
nav_order: 98
parent: 2022 Fall Seminar
author: 권기혁
# nav_exclude: true
---
# 3. Multivariate Linear Regression

$$ H(x_1, x_2, x_3) = x_1w_1 + x_2w_2 + x_3w_3 + b $$

$$ cost(W, b) = \frac{1}{m} \sum^m_{i=1} \left( H(x^{(i)}) - y^{(i)} \right)^2 $$

## Imports


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```


```python
# For reproducibility
torch.manual_seed(2)
```




    <torch._C.Generator at 0x7f93685faad0>



## Naive Data Representation


```python
# Data sample
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```


```python
# model Initialization
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 10000
for epoch in range(nb_epochs + 1):
    
    # H(x) 
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 
    cost = torch.mean((hypothesis - y_train) ** 2)

    # update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print log
    if epoch % 1000 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
```

    Epoch    0/10000 w1: 0.294 w2: 0.294 w3: 0.297 b: 0.003 Cost: 29661.800781
    Epoch 1000/10000 w1: 0.718 w2: 0.613 w3: 0.680 b: 0.009 Cost: 1.079378
    Epoch 2000/10000 w1: 0.757 w2: 0.571 w3: 0.682 b: 0.011 Cost: 0.754389
    Epoch 3000/10000 w1: 0.788 w2: 0.541 w3: 0.682 b: 0.012 Cost: 0.562648
    Epoch 4000/10000 w1: 0.812 w2: 0.517 w3: 0.681 b: 0.013 Cost: 0.448554
    Epoch 5000/10000 w1: 0.832 w2: 0.500 w3: 0.678 b: 0.014 Cost: 0.379734
    Epoch 6000/10000 w1: 0.848 w2: 0.488 w3: 0.675 b: 0.015 Cost: 0.337359
    Epoch 7000/10000 w1: 0.861 w2: 0.478 w3: 0.671 b: 0.016 Cost: 0.310472
    Epoch 8000/10000 w1: 0.871 w2: 0.472 w3: 0.667 b: 0.018 Cost: 0.292709
    Epoch 9000/10000 w1: 0.880 w2: 0.467 w3: 0.663 b: 0.019 Cost: 0.280348
    Epoch 10000/10000 w1: 0.888 w2: 0.464 w3: 0.658 b: 0.020 Cost: 0.271221
    

## Better way: Matrix Data Representation

$$
\begin{pmatrix}
x_1 & x_2 & x_3
\end{pmatrix}
\cdot
\begin{pmatrix}
w_1 \\
w_2 \\
w_3 \\
\end{pmatrix}
=
\begin{pmatrix}
x_1w_1 + x_2w_2 + x_3w_3
\end{pmatrix}
$$

$$ H(X) = XW $$


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```


```python
print(x_train.shape)
print(y_train.shape)
```

    torch.Size([5, 3])
    torch.Size([5, 1])
    


```python
# Initialization
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 10000
for epoch in range(nb_epochs + 1):
    
    # H(x)
    hypothesis = x_train.matmul(W) + b # or .mm or @

    # cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    # Update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print log
    if epoch % 1000 == 0:
        print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
            epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
        ))
```

    Epoch    0/10000 hypothesis: tensor([0., 0., 0., 0., 0.]) Cost: 29661.800781
    Epoch 1000/10000 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079375
    Epoch 2000/10000 hypothesis: tensor([152.1425, 184.1316, 180.7716, 196.8915, 140.7322]) Cost: 0.754390
    Epoch 3000/10000 hypothesis: tensor([151.9242, 184.2831, 180.7070, 196.8267, 140.9463]) Cost: 0.562648
    Epoch 4000/10000 hypothesis: tensor([151.7595, 184.3978, 180.6588, 196.7742, 141.1117]) Cost: 0.448561
    Epoch 5000/10000 hypothesis: tensor([151.6356, 184.4845, 180.6230, 196.7311, 141.2401]) Cost: 0.379736
    Epoch 6000/10000 hypothesis: tensor([151.5428, 184.5498, 180.5968, 196.6953, 141.3400]) Cost: 0.337359
    Epoch 7000/10000 hypothesis: tensor([151.4737, 184.5989, 180.5777, 196.6651, 141.4182]) Cost: 0.310472
    Epoch 8000/10000 hypothesis: tensor([151.4225, 184.6355, 180.5640, 196.6392, 141.4799]) Cost: 0.292709
    Epoch 9000/10000 hypothesis: tensor([151.3851, 184.6628, 180.5545, 196.6168, 141.5288]) Cost: 0.280348
    Epoch 10000/10000 hypothesis: tensor([151.3581, 184.6828, 180.5482, 196.5970, 141.5680]) Cost: 0.271221
    

## High-level Implementation with `nn.Module`

Do you remember this model?


```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```

We just need to change the input dimension from 1 to 3!


```python
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
```


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = MultivariateLinearRegressionModel()
print(list(model.parameters()))

optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 10000
for epoch in range(nb_epochs+1):
    
    prediction = model(x_train)
    
    cost = F.mse_loss(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

    [Parameter containing:
    tensor([[-0.1270,  0.4721,  0.0385]], requires_grad=True), Parameter containing:
    tensor([0.2394], requires_grad=True)]
    Epoch    0/10000 Cost: 19396.238281
    Epoch 1000/10000 Cost: 4.663135
    Epoch 2000/10000 Cost: 2.821095
    Epoch 3000/10000 Cost: 1.747797
    Epoch 4000/10000 Cost: 1.121640
    Epoch 5000/10000 Cost: 0.755653
    Epoch 6000/10000 Cost: 0.541065
    Epoch 7000/10000 Cost: 0.414593
    Epoch 8000/10000 Cost: 0.339471
    Epoch 9000/10000 Cost: 0.294261
    Epoch 10000/10000 Cost: 0.266545
    


```python
# Test set
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# prediction using the trained model
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
print(list(model.parameters()))

```

    훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[151.2622]], grad_fn=<AddmmBackward0>)
    [Parameter containing:
    tensor([[0.8220, 0.4362, 0.7517]], requires_grad=True), Parameter containing:
    tensor([-0.0155], requires_grad=True)]
    


```python

```


```python

```
