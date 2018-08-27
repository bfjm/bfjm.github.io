---
layout: post
title: torch训练分类器
date: 2018-08-28 05:00 +0800
categories: ml
tags: torch
---  
步骤：  
加载CIFAR10测试和训练数据集并规范化torchvision  
定义一个卷积神经网络  
定义一个损失函数  
在训练数据上训练网络  
在测试数据上测试网络  
{% highlight python linenos %}
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision数据集的输出范围是[0,1]的PILImage图像，我们将它们转换为归一化范围[-1,1]的张量

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
                                        download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                        download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')



import matplotlib.pyplot as plt
import numpy as np

#定义函数显示图像
def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))


#得到一些随机的训练图像
dataiter = iter(trainloader)
images,labels = dataiter.next()

#显示图像
imshow(torchvision.utils.make_grid(images))
#输出类别
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 定义一个卷积神经网络
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
#定义损失函数和优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#训练网络,2次
for epoch in range(2):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        #得到输入数据
        inputs,labels = data
        #包装数据
        inputs,labers = Variable(inputs), Variable(labels)
        #梯度清零
        optimizer.zero_grad()
        #forward+backward+optimize
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]

        #打印信息
        running_loss += loss.data[0]
        if i%2000 == 1999:
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0

print('Finished Training')

#在测试数据上测试网络

dataiter = iter(testloader)
images,labels = dataiter.next()

# 打印图像
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(Variable(images))
#输出的是10个类别的能量. 一个类别的能量越高, 则可以理解为网络认为越多的图像是该类别的. 那么, 让我们得到最高能量的索引:
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 数据在网络上执行
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


#表现良好与表现不佳
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
{% endhighlight %}