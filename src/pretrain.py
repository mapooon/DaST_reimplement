import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from models.Classifier import get_model
from PIL import Image


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


my_transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.1307,), (0.3081,))
     ])

train_loader=torch.utils.data.DataLoader(
    datasets.MNIST("data",train=True,download=True,transform=my_transform),batch_size=128,shuffle=True)
test_loader=torch.utils.data.DataLoader(
    datasets.MNIST("data",train=False,download=True,transform=my_transform),batch_size=128,shuffle=False
)
attack_test_loader=torch.utils.data.DataLoader(
    datasets.MNIST("data",train=False,download=True,transform=my_transform),batch_size=1,shuffle=False
)
print(len(train_loader))
print(len(test_loader))

model_name='cnn3'

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn=get_model(model_name,10,1).to(device)
cnn.train(mode=True)


criterion=nn.CrossEntropyLoss()#torch.nn.NLLLoss()



def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).detach().cpu().numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)
def train(m,out_dir):
    iter_loss=[]
    train_losses=[]
    test_losses=[]
    # iter_loss_path=os.path.join(out_dir,"iter_loss.csv")
    # epoch_loss_path=os.path.join(out_dir,"epoch_loss.csv")
    nb_epochs=5
    last_loss=99999
    mkdirs(os.path.join(out_dir,"models"))
    optimizer=optim.SGD(m.parameters(),lr=0.03,momentum=0.9)
    for epoch in range(nb_epochs):
        train_loss=0.
        train_acc=0.
        m.train(mode=True)
        for data,target in train_loader:
            data,target=data.to(device),target.to(device)
            optimizer.zero_grad()
            output=m(data)
            loss=criterion(output,target)
            loss_value=loss.item()
            iter_loss.append(loss_value)
            train_loss+=loss_value
            loss.backward()
            optimizer.step()
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            train_acc+=acc
        train_losses.append(train_loss/len(train_loader))
        
        test_loss=0.
        test_acc=0.
        m.train(mode=False)
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=m(data)
            loss=criterion(output,target)
            loss_value=loss.item()
            iter_loss.append(loss_value)
            test_loss+=loss_value
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            test_acc+=acc
        test_losses.append(test_loss/len(test_loader))
        print("Epoch {}: train loss is {}, train accuracy is {}, test loss is {}, test accuracy is {}".
              format(epoch,round(train_loss/len(train_loader),2),
                     round(train_acc/len(train_loader),2),
                     round(test_loss/len(test_loader),2),
                     round(test_acc/len(test_loader),2)))        
        if test_loss/len(test_loader)<last_loss:      
            save_model_path=os.path.join(out_dir,"models","best_model.pth".format(epoch))
            # torch.save({
            #         "model":m.state_dict(),
            #         "optimizer":optimizer.state_dict()
            #     },save_model_path)
            torch.save(m.state_dict(), save_model_path)
            last_loss=test_loss/len(test_loader)
        
    


train(cnn,model_name)


