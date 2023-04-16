'''---Import Libs---'''
import torch
import os
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv
import random

'''---Load Data From Local---'''
list_Data_Mask = []
list_Data_Nomask = []

#使用列表记录图像的路径、特征和target
for i in os.listdir('CMFD'):
    for j in os.listdir(f'CMFD/{i}'):    #遍历CMFD
        list_Data_Mask.append((f'CMFD/{i}/{j}',j[6:10],0))     #加入列表，元素的是图片存储地址
#print(CMFD)

for i in os.listdir('IMFD'):
    for j in os.listdir(f'IMFD/{i}'):    #遍历IMFD
        if (j[6:-4]=='Mask_Mouth_Chin'):
            list_Data_Nomask.append((f'IMFD/{i}/{j}',j[6:-4],1))
        elif  (j[6:-4]=='Mask_Nose_Mouth'):
            list_Data_Nomask.append((f'IMFD/{i}/{j}',j[6:-4],2))
        elif  (j[6:-4]=='Mask_Chin'):
            list_Data_Nomask.append((f'IMFD/{i}/{j}',j[6:-4],3))   #加入列表，元素的是图片存储地址
#print(CMFD)

for i in os.listdir('NMFD'):
    for j in os.listdir(f'NMFD/{i}'):    #遍历NMFD
        list_Data_Nomask.append((f'NMFD/{i}/{j}','NoMask',4))     #将没戴口罩的图片也存入列表中
#print(CMFD)



'''---creat a custom Dataset---'''
class MaskDataSet(Dataset):
        def __init__(self,path):
        #self.path = path
            self.labels = pd.read_csv(path)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            img_nparr = plt.imread(self.labels.iloc[index,1])  #这里返回tensor型1024x1024x3(H,W,C)的图像
            img_chw = np.transpose(img_nparr)         #此处进行转换
            img = torch.FloatTensor(img_chw)
            tar = torch.tensor(self.labels.iloc[index,3])
            return img,tar                       #这里返回tensor型3x1024x1024(C,H,W)的图像数据和对应的target

'''---load the train&test data from dataset---'''

def train_test_dataload():

    list_Train_Mask,list_Test_Mask = train_test_split(list_Data_Mask,train_size=0.025,test_size=0.005)

    list_Train_Nomask,list_Test_Nomask = train_test_split(list_Data_Nomask,train_size=0.025,test_size=0.005)

    list_Train = list_Train_Mask + list_Train_Nomask
    list_Test = list_Test_Mask + list_Test_Nomask

#创建DataFrame
    df_Train = pd.DataFrame(list_Train,columns=['Addr','Feature','Target'])
    df_Test = pd.DataFrame(list_Test,columns=['Addr','Feature','Target'])
#print(df_NOMASK)

#将DataFrame转换为.csv文件，方便在DataSet中读取和操作

    df_Train.to_csv('train.csv')
    df_Test.to_csv('test.csv')

#重写DataSet类，特别是其中的__len__和__getitem__方法
    
    dataset_Train = MaskDataSet('train.csv',)
    dataset_Test = MaskDataSet('test.csv')
    return DataLoader(dataset_Train,1,shuffle=True,drop_last=True),DataLoader(dataset_Test,1,shuffle=True,drop_last=True)   
    #此处确定一个超参数batch_size ,由于本机GPU内存太小，只能接受3个图像



'''---Build the CNN Model'''
def conv_block(in_channel, out_channel, kernel_size=3, strid=1, groups=1):
    padding = 0 if kernel_size == 1 else 1
    return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size,strid,padding=padding,groups=groups,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):  # 定义倒置残差结构,Inverted Residual
    def __init__(self, in_channel, out_channel, strid, t=6):  # 初始化方法
        super(InvertedResidual, self).__init__()  # 继承初始化方法
        self.in_channel = in_channel  # 输入通道数
        self.out_channel = out_channel  # 输出通道数
        self.strid = strid  # 步长
        self.t = t  # 中间层通道扩大倍数，对应原文expansion ratio
        self.hidden_channel = in_channel * t  # 计算中间层通道数
 
        layers = []  # 存放模型结构
        if self.t != 1:  # 如果expansion ratio不为1
            layers += [conv_block(self.in_channel, self.hidden_channel, kernel_size=1)]  # 添加conv+bn+relu6
        layers += [conv_block(self.hidden_channel, self.hidden_channel, strid=self.strid, groups=self.hidden_channel),
                   # 添加conv+bn+relu6，此处使用组数等于输入通道数的分组卷积实现depthwise conv
                   conv_block(self.hidden_channel, self.out_channel, kernel_size=1)[
                   :-1]]  # 添加1x1conv+bn，此处不再进行relu6
        self.residul_block = nn.Sequential(*layers)  # 倒置残差结构块
 
    def forward(self, x):  # 前传函数
        if self.strid == 1 and self.in_channel == self.out_channel:  # 如果卷积步长为1且前后通道数一致，则连接残差边
            return x + self.residul_block(x)  # x+F(x)
        else:  # 否则不进行残差连接
            return self.residul_block(x)  # F(x)
 


class MASKmodel(nn.Module):
    def __init__(self,num_classes):
        super(MASKmodel,self).__init__()

        self.num_classes = num_classes  # 类别数量
        self.feature = nn.Sequential(  # 特征提取部分
            conv_block(3, 32, strid=2),  # conv+bn+relu6,(n,3,224,224)-->(n,32,112,112)
            InvertedResidual(32, 16, strid=1, t=1),  # inverted residual block,(n,32,112,112)-->(n,16,112,112)
            InvertedResidual(16, 24, strid=2),  # inverted residual block,(n,16,112,112)-->(n,24,56,56)
            InvertedResidual(24, 24, strid=1),  # inverted residual block,(n,24,56,56)-->(n,24,56,56)
            InvertedResidual(24, 32, strid=2),  # inverted residual block,(n,24,56,56)-->(n,32,28,28)
            InvertedResidual(32, 32, strid=1),  # inverted residual block,(n,32,28,28)-->(n,32,28,28)
            InvertedResidual(32, 32, strid=1),  # inverted residual block,(n,32,28,28)-->(n,32,28,28)
            InvertedResidual(32, 64, strid=2),  # inverted residual block,(n,32,28,28)-->(n,64,14,14)
            InvertedResidual(64, 64, strid=1),  # inverted residual block,(n,64,14,14)-->(n,64,14,14)
            InvertedResidual(64, 64, strid=1),  # inverted residual block,(n,64,14,14)-->(n,64,14,14)
            InvertedResidual(64, 64, strid=1),  # inverted residual block,(n,64,14,14)-->(n,64,14,14)
            InvertedResidual(64, 96, strid=1),  # inverted residual block,(n,64,14,14)-->(n,96,14,14)
            InvertedResidual(96, 96, strid=1),  # inverted residual block,(n,96,14,14)-->(n,96,14,14)
            InvertedResidual(96, 96, strid=1),  # inverted residual block,(n,96,14,14)-->(n,96,14,14)
            InvertedResidual(96, 160, strid=2),  # inverted residual block,(n,96,14,14)-->(n,160,7,7)
            InvertedResidual(160, 160, strid=1),  # inverted residual block,(n,160,7,7)-->(n,160,7,7)
            InvertedResidual(160, 160, strid=1),  # inverted residual block,(n,160,7,7)-->(n,160,7,7)
            InvertedResidual(160, 320, strid=1),  # inverted residual block,(n,160,7,7)-->(n,320,7,7)
            conv_block(320, 1280, kernel_size=1)  # conv+bn+relu6,(n,320,7,7)-->(n,1280,7,7)
        )
 
        self.classifier = nn.Sequential(  # 分类部分
            nn.AdaptiveAvgPool2d(1),  # avgpool,(n,1280,7,7)-->(n,1280,1,1)
            nn.Conv2d(1280, self.num_classes, 1, 1, 0)  # 1x1conv,(n,1280,1,1)-->(n,num_classes,1,1),等同于linear
        )
 
    def forward(self, x):  # 前传函数
        x = self.feature(x)  # 提取特征
        x = self.classifier(x)  # 分类
        return x.view(-1, self.num_classes)



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        #_,pred=torch.max(outputs.data,1)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 300 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Result: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




device = torch.device("cuda:0")
model = MASKmodel(num_classes=5)
model.to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.002,momentum=0.8)#,weight_decay=0.8,nesterov=True)  
#print(model)



epochs = 10
for t in range(epochs):
    dataload_Data_train,dataload_Data_test = train_test_dataload()
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataload_Data_train, model, cost, optimizer)
    test_loop(dataload_Data_test, model, cost)
# Save the model as a file
print("Done!")