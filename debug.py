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

def train_test_dataload():

    list_Train_Mask,list_Test_Mask = train_test_split(list_Data_Mask,train_size=0.1,test_size=0.1)

    list_Train_Nomask,list_Test_Nomask = train_test_split(list_Data_Nomask,train_size=0.01,test_size=0.1)

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


    dataset_Train = MaskDataSet('train.csv')
    dataset_Test = MaskDataSet('test.csv')



    return DataLoader(dataset_Train,4,shuffle=True),DataLoader(dataset_Test,shuffle=True)   #此处确定一个超参数batch_size ,由于本机GPU内存太小，只能接受3个图像




class MASKmodel(nn.Module):
    def __init__(self):
        super(MASKmodel,self).__init__()

        self.flatten = nn.Flatten()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels = 30,kernel_size = 4,stride = 1,groups=1),  #这里输入的图像数据要是（C,H,W）
            nn.ReLU(),
            nn.MaxPool2d(4,4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels = 100,kernel_size = 4,stride = 1,groups=1),  #这里输入的图像数据要是（C,H,W）
            nn.ReLU(),
            nn.MaxPool2d(4,4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=100,out_channels = 300,kernel_size = 4,stride = 1,groups=1),  #这里输入的图像数据要是（C,H,W）
            nn.ReLU(),
            nn.MaxPool2d(4,4)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=300,out_channels = 500,kernel_size = 4,stride = 1,groups=1),  #这里输入的图像数据要是（C,H,W）
            nn.ReLU(),
            nn.MaxPool2d(4,4)
        )

        self.fc1 = nn.Linear(500*3*3,500)

        self.fc2 = nn.Linear(500,120)

        self.fc3 = nn.Linear(120,5)
    
    def forward(self,x):
        #x = self.flatten(x)
        ###卷积
        x=self.conv1(x)       #size:3x30x254x254
        x=self.conv2(x)       #size:3x100x62x62
        x=self.conv3(x)       #size:3x300x15x15
        x=self.conv4(x)       #size:3x500x3x3
        ###全连接
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,1)
        return  x             #size:




device = torch.device("cuda:0")
model = MASKmodel()
model.to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.7,momentum=0.8,weight_decay=0.8,nesterov=True)  #根据Loss函数的值的变化，调整超参数
#print(model)

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
        loss = cost(outputs,y)           ###训练时Loss不下降，可能是debug模式下，batch数据太少，还可能是，模型层数太少...
        
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += torch.sum(pred == y)
    testing_correct = 0   
    '''
    for data in dataload_Data_test:   
        outputs = model(X)
        _,pred=torch.max(outputs.data,1)
        testing_correct += torch.sum(pred == y)
    '''
    print("Loss is:{:.4f},Train Accuracy is:{:.4f}%,{}/{},Test Accuracy is:{:.4f}"\
          .format(running_loss/len(dataload_Data_train)\
          ,0.25*100*running_correct/len(dataload_Data_train)\
          ,running_correct
          ,4*len(dataload_Data_train)
          ,100*testing_correct/len(dataload_Data_test)))