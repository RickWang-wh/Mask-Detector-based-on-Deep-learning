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

    list_Train_Mask,list_Test_Mask = train_test_split(list_Data_Mask,train_size=0.05,test_size=0.05)

    list_Train_Nomask,list_Test_Nomask = train_test_split(list_Data_Nomask,train_size=0.05,test_size=0.05)

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
    return DataLoader(dataset_Train,3,shuffle=True),DataLoader(dataset_Test,3,shuffle=True)   #此处确定一个超参数batch_size ,由于本机GPU内存太小，只能接受3个图像



'''---Build the CNN Model'''
class MASKmodel(nn.Module):
    def __init__(self):
        super(MASKmodel,self).__init__()

        self.flatten = nn.Flatten(start_dim=1)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels = 32,kernel_size = 3,stride = 1),  #这里输入的图像数据要是（C,H,W）
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),

            nn.Conv2d(in_channels=32,out_channels = 64,kernel_size = 3,stride = 1),  #这里输入的图像数据要是（C,H,W）
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),

            nn.Conv2d(in_channels=64,out_channels = 128,kernel_size = 3,stride = 1),  #这里输入的图像数据要是（C,H,W）
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),

            nn.Conv2d(in_channels=128,out_channels = 256,kernel_size = 3,stride = 1),  #这里输入的图像数据要是（C,H,W）
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256,5),
            nn.Softmax(dim=1)

        )

    def forward(self,x):
        #x = self.flatten(x)
        ###卷积
        x = self.features(x)
        ###全连接
        x = self.flatten(x)
        x = self.classifier(x)
        return  x             #size:


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
model = MASKmodel()
model.to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)#,weight_decay=0.8,nesterov=True)
#print(model)



epochs = 10
for t in range(epochs):
    dataload_Data_train,dataload_Data_test = train_test_dataload()
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataload_Data_train, model, cost, optimizer)
    test_loop(dataload_Data_test, model, cost)
print("Done!")


'''
n_epochs = 5
#with torch.no_grad():
for epoch in range(n_epochs):
    dataload_Data_train,dataload_Data_test = train_test_dataload()
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch+1,n_epochs))
    print("-"*10)
    for batchs,(X,y) in enumerate(dataload_Data_train):
        X = X.to(device)
        y = y.to(device)
        outputs = model(X)
        _,pred=torch.max(outputs.data,1)
        optimizer.zero_grad()
        loss = cost(outputs,y)           
        
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += torch.sum(pred == y)


    testing_correct = 0   

    for batchs,(X,y) in enumerate(dataload_Data_test):
        X = X.to(device)
        y = y.to(device)   
        outputs = model(X)
        _,pred=torch.max(outputs.data,1)
        testing_correct += torch.sum(pred == y)

    print("Loss is:{:.4f},Train Accuracy is:{:.4f}%,{}/{},Test Accuracy is:{:.4f}%,{}/{}"\
          .format(running_loss/len(dataload_Data_train)\
          ,100*running_correct/len(dataload_Data_train)/3\
          ,running_correct\
          ,3*len(dataload_Data_train)\
          ,100*testing_correct/len(dataload_Data_test)/3\
          ,testing_correct\
          ,3*len(dataload_Data_test)))
'''