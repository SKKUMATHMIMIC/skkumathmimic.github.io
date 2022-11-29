---
layout: default
title:  "1_CNN warming up"
date: 2022-11-22 01:11:34 +0900
collection: 2022-2
usemathjax: true
last_modified_date: 2022-11-14 +0900
nav_order: 100
parent: 2022 Fall Seminar
author: 권기혁
# nav_exclude: true
---
## Basics

one layer consists of 

conv (nn.Conv2d) + Activation function (nn.ReLU) + maxpooling (nn.MaxPoold2d)

The model consists of three layers

Layer 1 : Convolutional layer:
conv (in_channel = 1, out_channel = 32, kernel_size=3, stride=1, padding=1) +  ReLU
maxpooling (kernel_size=2, stride=2))

Layer 2: Convolutional layer:
conv (in_channel = 32, out_channel = 64, kernel_size=3, stride=1, padding=1) + ReLU
maxpooling(kernel_size=2, stride=2))

Layer 3: Fully-Connected layer:
batch_size × 7 × 7 × 64 → batch_size × 3136

Neuron 10 with Softmax


```python
import torch
import torch.nn as nn
```


```python
# Set a tensor with 1 × 1 × 28 × 28
# batch size × channel × height × widht
inputs = torch.Tensor(1, 1, 28, 28)
print('Tensor shape : {}'.format(inputs.shape))
```

    Tensor shape : torch.Size([1, 1, 28, 28])
    

Layer 1: input is 1 channel and output 32 channels with kernel 3 and padding 1 


```python
conv1 = nn.Conv2d(1, 32, 3, padding=1)
print(conv1)
#(n+2p-f)/s + 1
```

    Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    

Layer 2: input 32 channel and output 64 channels with kernel 3 and padding 1 

####  if input image n*n, filter size f*f, padding = f, stride = s

#### -> output shape = (floor((n+2p-f)/s) + 1) * (floor((n+2p-f)/s) + 1)


```python
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)
#(n+2p-f)/s + 1
```

    Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    

Maxpooling: kernel = 2, stride = 2


```python
pool = nn.MaxPool2d(3, stride=2)
print(pool)
pool = nn.MaxPool2d(2)
print(pool)
```

    MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    

Let's connect them


```python
out = conv1(inputs)
print(out.shape)
```

    torch.Size([1, 32, 28, 28])
    

32 channel,  28 width, 28 heigth tensor

conv1의 out_channel은 32 -> 그래서 32 channel

padding = 1, kernel = 3 × 3 -> 28 width, 28 heigth -> 크기 보존

Now, maxpooling:


```python
out = pool(out)
print(out.shape)
```

    torch.Size([1, 32, 14, 14])
    

For the conv2


```python
out = conv2(out)
print(out.shape)
```

    torch.Size([1, 64, 14, 14])
    

64 channel,  14 width, 14 heigth tensor
conv1의 out_channel은 64 -> 그래서 64 channel
padding = 1, kernel = 3 × 3 -> 14 width, 14 heigth -> 크기 보존


```python
out = pool(out)
print(out.shape)
```

    torch.Size([1, 64, 7, 7])
    

현재 out의 크기는 1 × 64 × 7 × 7입니다. out의 첫번째 차원이 몇인지 출력


```python
out.size(0)
```




    1




```python
out.size(1)
```




    64



.view()를 사용하여 텐서를 펼치자


```python
# 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
out = out.view(out.size(0), -1) 
print(out.shape)
```

    torch.Size([1, 3136])
    

Fully-Connteced layer: output으로 10개의 뉴런을 사용하여 10개 차원의 텐서로 변환


```python
fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print(out.shape)
```

    torch.Size([1, 10])
    

## Now, Let's classify the MNIST dataset via CNNs.


```python

```
