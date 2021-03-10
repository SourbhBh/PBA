import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F 

class FeedFrwdNet(nn.Module):
    def __init__(self,ip_dim,hid_nrn,op_dim):
        self.ip_dim = ip_dim
        super(FeedFrwdNet,self).__init__()
        self.lin1 = nn.Linear(ip_dim,hid_nrn) #First hidden layer
        self.lin2 = nn.Linear(hid_nrn,op_dim) #Output layer with 10 neurons

    def forward(self,input):
        x = input.view(-1,self.ip_dim) #Sort of a numpy reshape function
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet,self).__init__()
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
