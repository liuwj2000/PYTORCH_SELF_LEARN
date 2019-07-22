import numpy as np
import torch
from torchvision.datasets import mnist

from torch import nn
from torch.autograd import Variable
import torchvision

#下载mnist数据集，因为我已经下载了，所以直接download=True
train_set=mnist.MNIST(
    './data',
    train=True,
    download=False)

test_set=mnist.MNIST(
    './data',
    train=False,
    download=False)

a_data,a_label=train_set[0]

#这时候的a_data是PIL库中的格式，我们需要把它转成ndarray的格式
a_data=np.array(a_data)
#print(a_data)
#输出如下,是每个元素 0~256 的28*28的图

#[[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   3.  18.  18.  18. 126. 136. 175.  26. 166. 255. 247. 127.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.  30.  36.  94. 154. 170. 253. 253. 253. 253. 253. 225. 172. 253. 242. 195.  64.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.  49. 238. 253. 253. 253. 253. 253. 253. 253. 253. 251.  93.  82.  82.  56.  39.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.  18. 219. 253. 253. 253. 253. 253. 198. 182. 247. 241.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.  80. 156. 107. 253. 253. 205.  11.   0.  43. 154.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.  14.   1. 154. 253.  90.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 139. 253. 190.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  11. 190. 253.  70.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  35. 241. 225. 160. 108.   1.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  81. 240. 253. 253. 119.  25.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  45. 186. 253. 253. 150.  27.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  16.  93. 252. 253. 187.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 249. 253. 249.  64.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  46. 130. 183. 253. 253. 207.   2.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  39. 148. 229. 253. 253. 253. 250. 182.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  24. 114. 221. 253. 253. 253. 253. 201.  78.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.  23.  66. 213. 253. 253. 253. 253. 198.  81.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.  18. 171. 219. 253. 253. 253. 253. 195.  80.   9.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.  55. 172. 226. 253. 253. 253. 253. 244. 133.  11.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0. 136. 253. 253. 253. 212. 135. 132.  16.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]]

#对于神经网络来说，我们第一层的输入就是28*28=784，所以我们需要将我们得到的数据进行交换，把它拉平成一维向量

def data_tf(x):
    x=np.array(x)
    x=(x-0.5)/0.5 # 标准化
    x=x.reshape((-1))#-1会把它所有的维度拉平成一维（所有其他的维度乘起来）
    return x


#重新导入数据集
train_set2=mnist.MNIST(
    './data',
    train=True,
    transform=torchvision.transforms.Compose(
         [torchvision.transforms.ToTensor(),
          data_tf]),
    #Compose将多个函数合并起来，对数据集进行改造
    #totensor将数据缩放到0~1
    #data_tf将数据变成标准正态分布
    download=False)

test_set2=mnist.MNIST(
    './data',
    train=False,
    transform=torchvision.transforms.Compose(
         [torchvision.transforms.ToTensor(),
          data_tf]),
    download=False)

#这样出来的数据就是每个784+label

#接下来创建mini-batch的迭代器
from torch.utils.data import DataLoader
train_data=DataLoader(train_set2,batch_size=64,shuffle=True)
test_data=DataLoader(test_set2,batch_size=128,shuffle=False)
#batch_size 批处理的数量
#shuffle 是否打乱

a,a_label=next(iter(train_data))
a_label=Variable(a_label)
#print(a.shape)
#64*784
#print(a_label.shape)
#784

#定义一个四层的神经网络
net=nn.Sequential(
    nn.Linear(28*28,400),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(400,200),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(200,100),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(100,10)
    )

#print(net)

#Sequential(
#  (0): Linear(in_features=784, out_features=400, bias=True)
#  (1): ReLU()
#  (2): Linear(in_features=400, out_features=200, bias=True)
#  (3): ReLU()
#  (4): Linear(in_features=200, out_features=100, bias=True)
#  (5): ReLU()
#  (6): Linear(in_features=100, out_features=10, bias=True)
#)

#定义损失函数和优化函数
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.8)

#开始训练
losses=[]
accuracy=[]
eval_losses=[]
eval_accuracy=[]
 
for e in range(20):
    train_loss=0
    train_accuracy=0
    net.train()#现在的神经网络是训练模式（可加可不加，建议加上）

    for input,label in train_data:
        input=Variable(input)
        label=Variable(label)

        output=net(input)
        loss=loss_func(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss=train_loss+loss.item()#记录误差

        #记录准确率
        _,prediction=output.max(1)
        #output是每一个元素分别属于十个数字的概率，max()后返回两个数，第一个是最大的那个数值，第二个是最大的数值所在的位置（行/列）
        #0表示每一行的最大的一个，1表示每一列的最大的一个
        #出来的也是tensor
        num_correct=(prediction==label).sum().item()#所有对的数量
        acc=num_correct/input.shape[0]
        train_accuracy+=acc

    losses.append(train_loss/len(train_data))
    accuracy.append(train_accuracy/len(train_data))

    #在测试集上检测结果
    eval_loss=0
    eval_acc=0
    net.eval()#进入预测模式
    for input,label in test_data:
        input=Variable(input)
        label=Variable(label)
        output=net(input)
        loss=loss_func(output,label)

        eval_loss+=loss.item()
        _,prediction=output.max(1)
        num_correct=(prediction==label).sum().item()#所有对的数量
        acc=num_correct/input.shape[0]
        eval_acc+=acc
    eval_losses.append(train_loss/len(train_data))
    eval_accuracy.append(train_accuracy/len(train_data))
    #print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
      #    .format(e, train_loss / len(train_data), train_accuracy / len(train_data), 
       #              eval_loss / len(test_data), eval_acc / len(test_data)))

#epoch: 0, Train Loss: 0.600816, Train Acc: 0.824011, Eval Loss: 0.335050, Eval Acc: 0.892108
#epoch: 1, Train Loss: 0.213414, Train Acc: 0.936251, Eval Loss: 0.170862, Eval Acc: 0.948972
#epoch: 2, Train Loss: 0.148294, Train Acc: 0.955274, Eval Loss: 0.157498, Eval Acc: 0.951048
#epoch: 3, Train Loss: 0.113509, Train Acc: 0.965385, Eval Loss: 0.099614, Eval Acc: 0.968157
#epoch: 4, Train Loss: 0.092557, Train Acc: 0.971598, Eval Loss: 0.094633, Eval Acc: 0.971519
#epoch: 5, Train Loss: 0.077768, Train Acc: 0.975630, Eval Loss: 0.082922, Eval Acc: 0.975178
#epoch: 6, Train Loss: 0.066646, Train Acc: 0.979544, Eval Loss: 0.081213, Eval Acc: 0.973695
#epoch: 7, Train Loss: 0.056029, Train Acc: 0.982693, Eval Loss: 0.075418, Eval Acc: 0.975870
#epoch: 8, Train Loss: 0.049172, Train Acc: 0.984542, Eval Loss: 0.083180, Eval Acc: 0.974189
#epoch: 9, Train Loss: 0.042478, Train Acc: 0.986807, Eval Loss: 0.074687, Eval Acc: 0.977749
#epoch: 10, Train Loss: 0.038459, Train Acc: 0.987473, Eval Loss: 0.068791, Eval Acc: 0.979727
#epoch: 11, Train Loss: 0.032609, Train Acc: 0.989705, Eval Loss: 0.080438, Eval Acc: 0.976760
#epoch: 12, Train Loss: 0.029896, Train Acc: 0.990322, Eval Loss: 0.071930, Eval Acc: 0.975672
#epoch: 13, Train Loss: 0.025549, Train Acc: 0.991921, Eval Loss: 0.092645, Eval Acc: 0.974387
#epoch: 14, Train Loss: 0.022544, Train Acc: 0.992937, Eval Loss: 0.070379, Eval Acc: 0.980419
#epoch: 15, Train Loss: 0.019322, Train Acc: 0.993870, Eval Loss: 0.067896, Eval Acc: 0.980419
#epoch: 16, Train Loss: 0.017557, Train Acc: 0.994486, Eval Loss: 0.070048, Eval Acc: 0.980617
#epoch: 17, Train Loss: 0.013943, Train Acc: 0.995686, Eval Loss: 0.082605, Eval Acc: 0.977749
#epoch: 18, Train Loss: 0.013976, Train Acc: 0.995652, Eval Loss: 0.086771, Eval Acc: 0.975771
#epoch: 19, Train Loss: 0.011259, Train Acc: 0.996702, Eval Loss: 0.077245, Eval Acc: 0.979727

import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.title('train loss')
plt.plot(np.arange(len(losses)),losses)
plt.subplot(2,2,2)
plt.title('train acc')
plt.plot(np.arange(len(accuracy)),accuracy)
plt.subplot(2,2,3)
plt.title('text loss')
plt.plot(np.arange(len(eval_losses)),eval_losses)
plt.subplot(2,2,4)
plt.title('text acc')
plt.plot(np.arange(len(eval_accuracy)),eval_accuracy)
plt.show()
