
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DaST(nn.Module):
    def __init__(self, backbone, n_classes, input_dim=3):
        super(DaST,self).__init__()
        self.generator=
        
    
    def forward(self,x):
        N=len(x)
        for i in range(self.stack_conv):
            x=self.conv[i](x)

        x=F.adaptive_avg_pool2d(x,1).view(N,-1)
        x=self.linear(x)

        return x


def get_model(name,n_classes,input_dim):
    assert name in ['cnn3','cnn4','cnn5','vgg16','resnet18','resnet50']
    assert input_dim in [1,3]

    if name[:3]=='cnn':
        return CNN(n_classes,input_dim,int(name[3]))
    elif name == 'vgg16':
        return models.vgg16(pretrained=False,n_classes=n_classes)
    elif name == 'resnet18':
        return models.resnet18(pretrained=False,n_classes=n_classes)
    elif name == 'resnet50':
        return models.resnet50(pretrained=False,n_classes=n_classes)
    
        
        
    

