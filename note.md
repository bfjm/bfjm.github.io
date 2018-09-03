---
layout: page
title: note
permalink: /note/
---  

1. 
```
z = x*y
判断z是否溢出，
!x || z/x==y
无符号:
高n位全为0，不溢出
带符号：
高n为全为0或者全为1，且与低n位最高位相同，不溢出
1111 1000
加减不分无符号，带符号
除法：
带符号:
加上偏移量(2*k-1),在右移k位，在截断  
```  
2. 更新pip  
python -m pip  install --upgrade pip  

3. static_cast强制转换 double temp;static_cast<int>temp;    
内联函数inline,简单调用  

4. torch.nn 只支持小批量(mini-batches), 不支持一次输入一个样本, 即一次必须是一个 batch.  
例如, nn.Conv2d 的输入必须是 4 维的, 形如 nSamples x nChannels x Height x Width.  
如果你只想输入一个样本, 需要使用 input.unsqueeze(0) 将 batch_size 设置为 1.  
添加环境变量  
sudo ln -s /文件path/ /usr/bin/文件

代价函数  详解  
```
假设有训练样本(x, y)，模型为h，参数为θ。h(θ) = θTx（θT表示θ的转置）。

（1）概况来讲，任何能够衡量模型预测出来的值h(θ)与真实值y之间的差异的函数都可以叫做代价函数C(θ)，如果有多个样本，则可以将所有代价函数的取值求均值，记做J(θ)。因此很容易就可以得出以下关于代价函数的性质：

对于每种算法来说，代价函数不是唯一的；  
代价函数是参数θ的函数；  
总的代价函数J(θ)可以用来评价模型的好坏，代价函数越小说明模型和参数越符合训练样本(x, y)；  
J(θ)是一个标量；  
（2）当我们确定了模型h，后面做的所有事情就是训练模型的参数θ。那么什么时候模型的训练才能结束呢？这时候也涉及到代价函数，由于代价函数是用来衡量模型好坏的，我们的目标当然是得到最好的模型（也就是最符合训练样本(x, y)的模型）。因此训练参数的过程就是不断改变θ，从而得到更小的J(θ)的过程。理想情况下，当我们取到代价函数J的最小值时，就得到了最优的参数θ.  
例如，J(θ) = 0，表示我们的模型完美的拟合了观察的数据，没有任何误差。

选择代价函数时，最好挑选对参数θ可微的函数（全微分存在，偏导数一定存在） 
一个好的代价函数需要满足两个最基本的要求:能够评估模型的准确性，对参数θ可微。  
1. 在线性回归中，最常用的是均方误差  
m: 训练样本的个数  
hθ(x):用参数θ和x预测出来的y值；
y:标准答案
上角标:第i个样本  
2. 在逻辑回归中，最常用的是代价函数是交叉熵(Cross Entropy)，交叉熵是一个常见的代价函数，在神经网络中也会用到。  
```  

```
# 带正则化的代价函数
import numpy as np
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    reg = (learningRate / (2 * len(X))* np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg
#1. 当y=1的时候，第二项(1-y)log(1-h(x))等于0 
#2. 当y=0的时候，ylog(h(x))等于0
```
<li>{{ site.time }}</li> 
[test](http://www.github.com)