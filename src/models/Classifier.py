
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class CNN(nn.Module):
    def __init__(self, n_classes, input_dim=3, stack_conv=3, BatchNorm=SynchronizedBatchNorm2d):
        super(CNN,self).__init__()
        assert stack_conv in [3,4,5]
        feature_dim=128
        # BatchNorm=nn.BatchNorm2d
        conv=[]
        conv.append(nn.Sequential(
                nn.Conv2d(input_dim, 16, 3, padding=1, bias=False),
                BatchNorm(16),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
        conv.append(nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                BatchNorm(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))    
        conv.append(nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                BatchNorm(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))  
        for i in range(3,stack_conv):
            conv.append(nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                BatchNorm(64),
                nn.ReLU()
            ))
        self.conv=nn.ModuleList(conv)
        self.stack_conv=stack_conv

        self.linear=nn.Linear(64,n_classes)
        self._init_weight()
    
    def forward(self,x):
        N=len(x)
        for i in range(self.stack_conv):
            x=self.conv[i](x)

        x=F.adaptive_avg_pool2d(x,1).view(N,-1)
        x=self.linear(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_model(name,n_classes,input_dim,pretrained=False):
    assert name in ['cnn3','cnn4','cnn5','vgg16','resnet18','resnet50']
    assert input_dim in [1,3]
    cwd=os.getcwd()
    if name[:3]=='cnn':
        base_model=CNN(n_classes,input_dim,int(name[3]))
        if pretrained:
            base_model.load_state_dict(torch.load('src/models/'+name+'.pth'))
        return base_model
    elif name == 'vgg16':
        return models.vgg16(pretrained=False,n_classes=n_classes)
    elif name == 'resnet18':
        return models.resnet18(pretrained=False,n_classes=n_classes)
    elif name == 'resnet50':
        return models.resnet50(pretrained=False,n_classes=n_classes)
    
        
        
    

