---
layout: default
title:  "7_1_tips"
date: 2022-11-22 01:11:34 +0900
collection: 2022-2
usemathjax: true
last_modified_date: 2022-11-14 +0900
nav_order: 95
parent: 2022 Fall Seminar
author: 권기혁
# nav_exclude: true
---
# Tips


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




    <torch._C.Generator at 0x7fc269cceb10>



## Training and Test Datasets


```python
x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]
                            ])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
```


```python
x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])
print(x_test)
print(y_test)
```

    tensor([[2., 1., 1.],
            [3., 1., 2.],
            [3., 3., 4.]])
    tensor([2, 2, 2])
    

## Model


```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
    def forward(self, x):
        return self.linear(x)
```


```python
model = SoftmaxClassifierModel()
```


```python
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)
```


```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 500
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.cross_entropy(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```


```python
def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print('Accuracy: {}% Cost: {:.6f}'.format(
         correct_count / len(y_test) * 100, cost.item()
    ))
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/500 Cost: 2.203667
    Epoch   50/500 Cost: 0.846174
    Epoch  100/500 Cost: 0.720624
    Epoch  150/500 Cost: 0.650993
    Epoch  200/500 Cost: 0.605071
    Epoch  250/500 Cost: 0.571026
    Epoch  300/500 Cost: 0.543846
    Epoch  350/500 Cost: 0.521098
    Epoch  400/500 Cost: 0.501460
    Epoch  450/500 Cost: 0.484143
    


```python
test(model, optimizer, x_test, y_test)
```

    Accuracy: 100.0% Cost: 0.004228
    

## Learning Rate

Gradient Descent 에서의 $\alpha$ 값

`optimizer = optim.SGD(model.parameters(), lr=0.1)` 에서 `lr=0.1` 이다

learning rate이 너무 크면 diverge 하면서 cost 가 점점 늘어난다 (overshooting).


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e5)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/500 Cost: 1.280268
    Epoch   50/500 Cost: 1282693.125000
    Epoch  100/500 Cost: 896298.750000
    Epoch  150/500 Cost: 203640.000000
    Epoch  200/500 Cost: 43079.625000
    Epoch  250/500 Cost: 927507.812500
    Epoch  300/500 Cost: 416148.062500
    Epoch  350/500 Cost: 722382.250000
    Epoch  400/500 Cost: 679853.875000
    Epoch  450/500 Cost: 628648.062500
    

learning rate이 너무 작으면 cost가 거의 줄어들지 않는다.


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-10)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/500 Cost: 3.187324
    Epoch   50/500 Cost: 3.187324
    Epoch  100/500 Cost: 3.187324
    Epoch  150/500 Cost: 3.187324
    Epoch  200/500 Cost: 3.187324
    Epoch  250/500 Cost: 3.187324
    Epoch  300/500 Cost: 3.187324
    Epoch  350/500 Cost: 3.187324
    Epoch  400/500 Cost: 3.187324
    Epoch  450/500 Cost: 3.187324
    

적절한 숫자로 시작해 발산하면 작게, cost가 줄어들지 않으면 크게 조정하자.


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/500 Cost: 1.341574
    Epoch   50/500 Cost: 0.811143
    Epoch  100/500 Cost: 0.705258
    Epoch  150/500 Cost: 0.644916
    Epoch  200/500 Cost: 0.602896
    Epoch  250/500 Cost: 0.570596
    Epoch  300/500 Cost: 0.544228
    Epoch  350/500 Cost: 0.521857
    Epoch  400/500 Cost: 0.502380
    Epoch  450/500 Cost: 0.485115
    

## Data Preprocessing (데이터 전처리)

데이터를 zero-center하고 normalize하자.


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

$$ x'_j = \frac{x_j - \mu_j}{\sigma_j} $$

여기서 $\sigma$ 는 standard deviation, $\mu$ 는 평균값 이다.


```python
mu = x_train.mean(dim=0)
```


```python
sigma = x_train.std(dim=0)
```


```python
norm_x_train = (x_train - mu) / sigma
```


```python
print(norm_x_train)
```

    tensor([[-1.0674, -0.3758, -0.8398],
            [ 0.7418,  0.2778,  0.5863],
            [ 0.3799,  0.5229,  0.3486],
            [ 1.0132,  1.0948,  1.1409],
            [-1.0674, -1.5197, -1.2360]])
    

Normalize와 zero center한 X로 학습해서 성능을 보자


```python
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
```


```python
model = MultivariateLinearRegressionModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 100
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```


```python
train(model, optimizer, norm_x_train, y_train)
```

    Epoch    0/100 Cost: 29785.089844
    Epoch   10/100 Cost: 338.934174
    Epoch   20/100 Cost: 4.423987
    Epoch   30/100 Cost: 0.436581
    Epoch   40/100 Cost: 0.301752
    Epoch   50/100 Cost: 0.239829
    Epoch   60/100 Cost: 0.198096
    Epoch   70/100 Cost: 0.169735
    Epoch   80/100 Cost: 0.150454
    Epoch   90/100 Cost: 0.137347
    

## Overfitting

너무 학습 데이터에 한해 잘 학습해 테스트 데이터에 좋은 성능을 내지 못할 수도 있다.

이것을 방지하는 방법은 크게 세 가지인데:

1. 더 많은 학습 데이터
2. 더 적은 양의 feature
3. **Regularization**

Regularization: Let's not have too big numbers in the weights


```python
def train_with_regularization(model, optimizer, x_train, y_train):
    nb_epochs = 100
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)
        
        # l2 norm 계산
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
            
        cost += l2_reg

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch+1, nb_epochs, cost.item()))
```


```python
model = MultivariateLinearRegressionModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
train_with_regularization(model, optimizer, norm_x_train, y_train)
```

    Epoch    1/100 Cost: 29477.810547
    Epoch   11/100 Cost: 518.073608
    Epoch   21/100 Cost: 188.472290
    Epoch   31/100 Cost: 184.613403
    Epoch   41/100 Cost: 184.534485
    Epoch   51/100 Cost: 184.513443
    Epoch   61/100 Cost: 184.501404
    Epoch   71/100 Cost: 184.494324
    Epoch   81/100 Cost: 184.490158
    Epoch   91/100 Cost: 184.487701
    
