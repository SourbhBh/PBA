import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from nn_models import *

def get_accuracy(tr_proj,ts_proj,tr_labels,ts_labels,epochs):
    trainset = torch.utils.data.TensorDataset(torch.Tensor(np.array(tr_proj)), torch.Tensor(tr_labels))
    testset = torch.utils.data.TensorDataset(torch.Tensor(np.array(ts_proj)), torch.Tensor(ts_labels))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)
    ip_dim = torch.tensor(trainloader.dataset.tensors[0][0].size()).item()
    hid_nrn = 100
    op_dim = int(torch.max(trainloader.dataset.tensors[1]).item())+1
    model = FeedFrwdNet(ip_dim,hid_nrn,op_dim)
    #Fix Loss function
    loss = nn.CrossEntropyLoss()
    #Define Optimizer to update weights
    optimizer = optim.Adam(model.parameters(),lr=0.0005)
    for epoch in range(1, epochs + 1):
        for batch_idx, data in enumerate(trainloader,0):
            #data, target = Variable(data), Variable(target)
            inputs, labels = data
            #print(type(labels))
            #labels = int(labels)
            labels = labels.long()
            optimizer.zero_grad() #Always zero gradient buffers
            output = model(inputs)
            loss_v = loss(output, labels)
            loss_v.backward()
            optimizer.step()
            #if batch_idx % log_interval == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, batch_idx * len(inputs), len(trainloader.dataset),
            #        100. * batch_idx / len(trainloader), loss_v.data))       train(epoch,trainloader,optimizer,model,loss)
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        labels = labels.long()
        outputs = model(images)
        predicted = outputs.argmax(dim=1, keepdim=True)
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()

    print('Accuracy of the network on the %d test images: %f %%' % (total, 100.0 * correct / total))
    return 100.0*(correct/total)

class CustomTensorDataset(torch.utils.data.TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, arrays, transform=None):
        #assert all(arr[0].size(0) == arr.size(0) for arr in arrays)
        self.arrs = arrays
        self.transform = transform

    def __getitem__(self, index):
        x = self.arrs[0][index]
        #x = torch.transpose(x,2,1,0)
        if self.transform:
            x = self.transform(x)
        y = self.arrs[1][index]

        return x, y

    def __len__(self):
        return self.arrs[0].shape[0]

def cifar_get_accuracy(train_data, train_labels, test_data, test_labels,epochs):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CustomTensorDataset(arrays=(train_data,train_labels),transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True,num_workers=2)
    testset = CustomTensorDataset(arrays=(test_data,test_labels),transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False,num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = CIFARNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            labels = labels.long()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return 100*correct/total
