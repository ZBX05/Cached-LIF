import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from model import SNN
import argparse
from model import SNN
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader
import time
from dataset import *
from train import evaluate

import numpy as np
from spikingjelly.datasets import play_frame
import spikingjelly
from tqdm import tqdm

_,test_data_loader,shape=load_dataset_cifar10(True,r'F:\Temporal_test\CIFAR10\data',64,True)

model=SNN('CONVNP-64-3-1-1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-128-3=3-1=1-1=1_RES-256-3=3-2=1-1=1_RES-256-3=3-1=1-1=1_RES-256-3=3-1=1-1=1_RES-512-3=3-2=1-1=1_RES-512-3=3-1=1-1=1-_FC-256_L-10',2,shape,0,'BN',1.0,0.0,2.0,'triangle',1.0,True,False)
model.load_state_dict(torch.load(r'F:\Temporal_Regularization_Training\CIFAR10\result\CE_T2_triangle_tdBN_ResNet-19_2025-04-09_18-12-21\weights\CE_T2_triangle_tdBN_ResNet-19_triangle1.0_283_0.1840170666575432_0.9461.pth'))
model.eval()

neuron_status=None
model.to('cuda')
with torch.no_grad():
    for img,labels in tqdm(test_data_loader):
        img=img.to('cuda')
        labels=labels.to('cuda')
        _,layer_output=model.get_spike_firing_patterns(img,False)
        if neuron_status is None:
            neuron_status=[torch.amax(l,dim=(0,1)) for l in layer_output]
        else:
            layer_output=[torch.amax(l,dim=(0,1)) for l in layer_output]
            for i in range(len(neuron_status)):
                neuron_status[i]=(neuron_status[i]+layer_output[i]).gt(0).float()
neuron_status=[s.flatten() for s in neuron_status]
neuron_status=torch.cat(neuron_status,dim=0)
total_neuron=neuron_status.sum().item()
silent_neuron=neuron_status.eq(0).sum().item()
print(total_neuron,silent_neuron,silent_neuron/total_neuron)