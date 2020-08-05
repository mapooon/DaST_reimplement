import torch
from PIL import Image
from torchvision import models,datasets,transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from visdom import Visdom
import random
import sys
import shutil
import argparse
from models.Generator import DaSTGenerator
from models.Classifier import get_model
# from mmcv import Config
from utils.logs import log
from datetime import datetime
# from advertorch.attacks import GradientSignAttack
from utils.fgsm import fgsm_attack
import pandas as pd

# parser=argparse.ArgumentParser()
# # parser.add_argument('-s',dest='seed',default=None)
# parser.add_argument('-c',dest='config')
# # parser.add_argument('-w',dest='weight_name',default=None)
# # parser.add_argument('-b',dest='batch_size',default=4,type=int)
# # parser.add_argument('-e',dest='n_epoch',default=500,type=int)
# # parser.add_argument('-r',dest='use_resize',default=)
# args=parser.parse_args()

seed=1
#seed=1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda')

# cfg = Config.fromfile(args.config)

# parameter setting
batch_size=400#int(cfg.train_cfg.batch_size)
#n_epoch=cfg.train_cfg.n_epoch
n_query=10000000
n_epoch=500
n_step=n_query//batch_size//n_epoch#+(n_query%batch_size>0)

# model preparetion
n_channel=1# if cfg.dataset_type=='mnist' else 3
generatorP=DaSTGenerator(10,n_channel)
generatorL=DaSTGenerator(10,n_channel)
# attacked_model=get_model(cfg.att,10,n_channel)
# subL_model=get_model(cfg.sub)
# subP_model=get_model(cfg.sub)
attacked_model=get_model('cnn4',10,n_channel,pretrained=True)
subL_model=get_model('cnn5',10,n_channel)
subP_model=get_model('cnn5',10,n_channel)


generatorL = nn.DataParallel(generatorL).to(device)
generatorP = nn.DataParallel(generatorP).to(device)
subL_model = nn.DataParallel(subL_model).to(device)
subP_model = nn.DataParallel(subP_model).to(device)
attacked_model = nn.DataParallel(attacked_model).to(device)
sub_models={
    'L':subL_model,
    'P':subP_model
    }
gen_models={
    'L':generatorL,
    'P':generatorP
}

# if parsed.weight_name:
#     model.load_state_dict(torch.load(os.path.join(current,parsed.weight_name)))

opt_genL = optim.Adam(generatorL.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1e-5) 
opt_genP = optim.Adam(generatorP.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1e-5)
opt_subL = optim.Adam(subL_model.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1e-5) 
opt_subP = optim.Adam(subP_model.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1e-5) 
opt_sub={
    'L':opt_subL,
    'P':opt_subP
    }
opt_gen={
    'L':opt_genL,
    'P':opt_genP
    }

# for name, param in generatorP.named_parameters():
#     if param.requires_grad:
#         print(name)
# sys.exit()

# loss_dict={x:np.array([np.nan]*num_epochs) for x in ['train', "val"]}

# loss function
MSE=nn.MSELoss()
# L1=nn.L1Loss()
CE=nn.CrossEntropyLoss()

# output setting
now=datetime.now()
save_path='output/'+now.strftime("%m_%d_%H_%M_%S")+'/'
os.mkdir(save_path)
os.mkdir(save_path+'weights/')
os.mkdir(save_path+'logs/')
logger = log(path=save_path+"logs/", file="losses.logs")


running_loss = 0.0

my_transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.1307,), (0.3081,))
     ])
mnist_dataset=datasets.MNIST("data",train=False,download=True,transform=my_transform)
test_loader=torch.utils.data.DataLoader(
    mnist_dataset,batch_size=batch_size,shuffle=False
)


for data,target in test_loader:
    data,target=data.to(device),target.to(device)
    # data.requires_grad=True

    # output_sub=sub_models[phase](data)
    # acc_sub=(output_sub.argmax(1)==target)

    with torch.no_grad():
        output_att=attacked_model(data)
    acc_att=(output_att.argmax(1)==target).sum().item()/len(target)
    print(acc_att)

#     print(F.softmax(output_att,dim=1)[0].cpu().data.numpy().tolist())
#     sys.exit()


for phase in ['P','L']:
    LD_min=np.inf
    LG_min=np.inf
    acc=0
    asr=0
    flag=True
    step=0
    n_time=time.time()
    # df=pd.DataFrame()
    # df["Acc_mnist"]=[]
    # df['Acc_synth']=[]
    # df["ASR"]=[]
    # df['L_D']=[]
    # df['L_C']=[]
    acc_mnist_list=[]
    acc_synth_list=[]
    asr_list=[]
    l_d_list=[]
    l_c_list=[]
    # adversary = GradientSignAttack(
    # sub_models[phase], loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    # eps=0.3, clip_min=0.0, clip_max=1.0, targeted=False)
    for epoch in range(n_epoch):
        for step in range(n_step):
    # while flag:
        # step+=1
            sub_models[phase].zero_grad()
            gen_models[phase].zero_grad()
            inputs,targets=gen_models[phase](batch_size,device=device,shuffle=True)
            # print(targets)
            with torch.no_grad():
                outputs_att=attacked_model(inputs.detach())
            outputs_sub=sub_models[phase](inputs.detach())

            # print(inputs.shape,targets.shape,outputs_att.shape)

            if phase=='P':
                L_D=F.mse_loss(F.softmax(outputs_sub,dim=1),F.softmax(outputs_att,dim=1))#+CE(outputs_sub,outputs_att.argmax(1))
            else:
                L_D=CE(outputs_sub,outputs_att.argmax(1))

            

            L_D.backward()
            opt_sub[phase].step()
            sub_models[phase].zero_grad()
            gen_models[phase].zero_grad()

            outputs_sub=sub_models[phase](inputs)
            with torch.no_grad():
                outputs_att=attacked_model(inputs)
            if phase=='P':
                L_D=F.mse_loss(F.softmax(outputs_sub,dim=1),F.softmax(outputs_att,dim=1))#+CE(outputs_sub,outputs_att.argmax(1))
            else:
                L_D=CE(outputs_sub,outputs_att.argmax(1))

            L_C=CE(outputs_sub,targets.long())
            L_G=(-L_D).exp()+0.2*L_C
            L_G.backward()
            opt_gen[phase].step()
            
            acc_step=(outputs_sub.argmax(1).long()==targets.long()).long().sum().item()/len(targets)
            # print(outputs_sub.argmax(1)[:10])
            # print(targets[:10])
            # print((outputs_sub.argmax(1).long()==targets.long())[:10])
            # print((outputs_sub.argmax(1).long()==targets.long())[:10].sum())
            
            # if L_G.item()>=LG_min and L_D.item()>=LD_min:
            #     flag=False
            # else:
            #     LG_min=L_G.item()
            #     LD_min=L_D.item()

        sum_sample=0
        acc_adv_=0
        attack_success=0
        acc_sub=0
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            data.requires_grad=True

            output_sub=sub_models[phase](data)
            acc_sub+=(output_sub.argmax(1)==target).sum().item()
            

            with torch.no_grad():
                output_att=attacked_model(data)
            acc_att=(output_att.argmax(1)==target)

            if acc_att.sum().item()==0:
                continue
            sum_sample+=acc_att.sum().item()

            loss=CE(output_sub,target)
            sub_models[phase].zero_grad()
            loss.backward()

            data_grad=data.grad.data
            adv_data=fgsm_attack(data,data_grad,epsilon=0.3)
            output_adv=attacked_model(adv_data)
            acc_adv_+=(output_adv.argmax(1)==target).sum().item()
            attack_success+=((output_adv.argmax(1)!=target)*acc_att).sum().item()
        # print(output_sub.argmax(1)[:10])
        # print(output_att.argmax(1)[:10])
        # print(target[:10])

        acc_sub/=len(mnist_dataset)
        # acc_adv_step=acc_adv_/sum_sample
        asr_step=attack_success/sum_sample

        if acc_step<acc and asr_step<asr:
            # flag=False
            pass
        else:
            acc=acc_step#max(acc_step,acc)
            asr=asr_step

    
        # if step>=n_step:
        #     flag=False
        
        torch.cuda.empty_cache()
        iter_time=time.time()
        # logger.info('Iter: {} | L_D: {:.4f} | L_G: {:.4f} | Time: {} sec'.format(phase, step, L_D, L_G, int(iter_time-n_time)))
        # logger.info('Scenario: {} | Iter: {}/{} | Acc_synthe: {:.4f} ({}/{}) | Acc_mnist: {:.4f} | ASR: {:.4f} ({}/{}) | L_D: {:.4f} | L_C: {:.4f} | Time: {} sec'.format(phase, step, n_step,acc_step,(outputs_sub.argmax(1).long()==targets.long()).long().sum().item(),len(targets), acc_sub, asr_step,attack_success,sum_sample, L_D.item(), L_C.item(), int(iter_time-n_time)))
        logger.info('Scenario: {} | Iter: {}/{} | Acc_synthe: {:.4f} ({}/{}) | Acc_mnist: {:.4f} | ASR: {:.4f} ({}/{}) | L_D: {:.4f} | L_C: {:.4f} | Time: {} sec'.format(phase, epoch,n_epoch,acc_step,(outputs_sub.argmax(1).long()==targets.long()).long().sum().item(),len(targets), acc_sub, asr_step,attack_success,sum_sample, L_D.item(), L_C.item(), int(iter_time-n_time)))
        torch.save(sub_models[phase].state_dict(), os.path.join(save_path+'weights/',"{}_{}_{:.4f}.pth".format(phase,epoch,acc_sub)))
        acc_mnist_list.append(acc_sub)
        acc_synth_list.append(acc_step)
        asr_list.append(asr_step)
        l_d_list.append(L_D.item())
        l_c_list.append(L_C.item())
    df=pd.DataFrame()
    df['Acc_mnist']=acc_mnist_list
    df['Acc_synth']=acc_synth_list
    df['ASR']=asr_list
    df['L_D']=l_d_list
    df['L_C']=l_c_list
    df.to_csv(os.path.join(save_path+'{}_result.csv'.format(phase)))
    logger.info('-------------------------------------------------------------------------------------------------------------------------------------------------------')
# サンプル数で割って平均を求める
# epoch_loss = running_loss / (step+1)#dataset_sizes[phase]#dataset_sizes[phase]
# epoch_iou = running_iou / (step+1)#dataset_sizes[phase]#dataset_sizes[phase]

# print('{} Loss: {:.4f} IOU: {:.4f}'.format(phase, epoch_loss, epoch_iou))
#print('{} Loss: {:.4f} '.format(phase, epoch_loss))

# loss_dict[phase][epoch]=epoch_loss
# iou_dict[phase][epoch]=epoch_iou
#visdom

    

# print()

# time_elapsed = time.time() - since
# print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# print('Best val iou: {:.4f}'.format(best_iou))

