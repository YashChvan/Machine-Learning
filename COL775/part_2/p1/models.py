from torch import nn
from torch.utils.data import dataloader
import torch
import torchvision
import torchvision.transforms as transforms
import math
import torch.optim as optim
import regs
import csv
from sklearn.metrics import f1_score
import os
# from utils import progress_bar
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
def norm_layer(num_features,reg = "inbuilt"):
    reg_layer = nn.BatchNorm2d(num_features)
    if(reg == 'bn'):
        reg_layer = regs.BatchNorm(num_features)
        
    if(reg == 'ln'):
        reg_layer = regs.LayerNorm(num_features)
        
    if(reg == 'in'):
        reg_layer = regs.InstanceNorm(num_features)
        
    if(reg == 'bin'):
        reg_layer = regs.BatchInstanceNorm(num_features)
        
    if(reg == 'gn'):
        reg_layer = regs.GroupNorm(num_features)
        
    if(reg == 'nn'):
        reg_layer = regs.NoNorm()
    return reg_layer

class block(nn.Module):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)
    
    def __init__(self,in_channels,out_channels,stride=1,downsampling = None,reg = "inbuilt"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding="same",stride=1, bias=False)
        self.norm1 = norm_layer(out_channels,reg = reg)
        self.conv2 = nn.Conv2d(out_channels,out_channels,padding=1,kernel_size=3,stride=stride, bias=False)
        self.norm2 = norm_layer(out_channels,reg = reg)
        self.relu = nn.ReLU()
        self.downsamp = downsampling
        if(self.downsamp is None):
            self.downsamp = norm_layer(out_channels,"nn")
        self.stride = stride
        
    def forward(self,x):
        identity = x.clone()
        out = x.clone()
        # print(out.size())
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        identity = self.downsamp(x)
        out += identity
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self,n,r,reg = "pyt_bn"):
        super().__init__()
        cluster1 = []
        for i in range(n):
            cluster1.append(block(16,16))
        self.c1 = nn.Sequential(*cluster1)
        cluster2 = []

        down2 = self.downsample = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                norm_layer(32,reg = reg)
                )
        cluster2.append(block(16,32,stride=2,downsampling = down2,reg=reg))
        for i in range(n-1):
            cluster2.append(block(32,32))
        self.c2 = nn.Sequential(*cluster2)
        
        cluster3 = []
        down3= self.downsample = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                norm_layer(64,reg = reg)
                )
        cluster3.append(block(32,64,stride=2,downsampling = down3,reg=reg))
        for i in range(n-1):
            cluster3.append(block(64,64))
        self.c3 = nn.Sequential(*cluster3)
        self.conv0 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding="same", bias=False)
        self.norm0 = norm_layer(16,reg = reg)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.output = nn.Linear(64,r)

    def forward(self,x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.relu(out)
        out = self.c1(out)
        # out = self.down1(out)
        out = self.c2(out)
        # out = self.down2(out)
        out = self.c3(out)
        out = self.avg_pool(out)
        out = out.squeeze()
        out = self.output(out)
        return out

