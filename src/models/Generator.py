from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Generator(nn.Module):
    def __init__(self, BatchNorm, nz=100, nch_g=64, nch=3):
        super(Generator, self).__init__()
        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 8, 4, 1, 0, bias=False),     
                BatchNorm(nch_g * 8),                      
                nn.LeakyReLU(0.2, inplace=True),                                       
            ),  # (100, 1, 1) -> (512, 4, 4)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 8, nch_g * 4, 4, 2, 1, bias=False),
                BatchNorm(nch_g * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (512, 4, 4) -> (256, 8, 8)
            'layer2': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 4, 2, 1, bias=False),
                BatchNorm(nch_g * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (256, 8, 8) -> (128, 16, 16)
 
            'layer3': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1, bias=False),
                BatchNorm(nch_g),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (128, 16, 16) -> (64, 32, 32)
            # 'layer4': nn.Sequential(
            #     nn.ConvTranspose2d(nch_g, nch_g, 4, 2, 1),
            #     nn.Tanh()
            # )   # (64, 32, 32) -> (3, 64, 64)
        })
        self._init_weight()
        
 
    def forward(self, z):
        for layer in self.layers.values():  
            z = layer(z)
        return z

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


class DaSTGenerator(nn.Module):
    def __init__(self, n_classes,n_channel, BatchNorm=SynchronizedBatchNorm2d):
        super(DaSTGenerator, self).__init__()
        
        class_gen=[]
        for i in range(n_classes):
            class_gen.append(Generator(BatchNorm))
        self.class_gen=nn.ModuleList(class_gen)

        self.n_classes=n_classes

        self.conv=nn.Sequential(
                    nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3, padding =1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 3,padding = 1, bias=False),
                    nn.BatchNorm2d(128),
                    # nn.ReLU(),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels = 128,out_channels = 64,kernel_size = 3, padding =1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels = 64,out_channels = 32,kernel_size = 3, padding =1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels = 32,out_channels = n_channel,kernel_size = 3, padding =1, bias=True),
                    # nn.BatchNorm2d(n_channel),
                    nn.Sigmoid()
                    )

        self._init_weight()




    def forward(self, batch_size,device='cpu',shuffle=False,size=(28,28)):
        
        class_bs=batch_size//self.n_classes+((batch_size%self.n_classes)>0)

        z=torch.randn(class_bs, 100, 1, 1).to(device)
        outputs=self.class_gen[0](z)
        outputs=F.interpolate(outputs, size=size, mode='bilinear', align_corners=True)
        targets=torch.Tensor([0]*class_bs).to(device)
        for i in range(1,self.n_classes):
            z=torch.randn(class_bs, 100, 1, 1).to(device)
            output=self.class_gen[i](z)
            output=F.interpolate(output, size=size, mode='bilinear', align_corners=True)
            outputs=torch.cat([outputs,output],dim=0)
            targets=torch.cat([targets,torch.Tensor([i]*class_bs).to(device)],dim=0)
            
        outputs=self.conv(outputs)

        N=len(outputs)
        if shuffle:
            idx=np.random.permutation(range(N))
        else:
            idx=list(range(N))

        return outputs[idx],targets[idx]

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

    