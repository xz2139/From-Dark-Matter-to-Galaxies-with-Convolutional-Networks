import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim

from args import args
from train_f import *
from Dataset import Dataset

class Baseline(nn.Module):

    def __init__(self, in_ch, out_ch):
        
        super(Baseline, self).__init__()
        self.draft_model = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, cube):
        cube = self.draft_model(cube)
        return cube
    
class SimpleUnet(nn.Module):

    def __init__(self, in_channels, target_class):
        
        super(SimpleUnet, self).__init__()
        self.target_class = target_class

        self.in_channels = in_channels
        self.conv1=nn.Conv3d(self.in_channels, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2=nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool1=nn.AvgPool3d(2)
        self.conv3=nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool2=nn.AvgPool3d(2)
        self.conv4=nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0, bias=True)

        self.relu=nn.ReLU()
        if self.target_class == 0:
            self.fc3=nn.Linear(4096,32768*2)
        else:
            self.fc3=nn.Linear(4096,32768)
    
    
    def forward(self, cube):
        b_size=cube.size()[0]
        cube = self.conv1(cube)
        cube = self.relu(cube)
        cube = self.conv2(cube)
        cube = self.relu(cube)
        cube = self.pool1(cube)

        cube = self.conv3(cube)
        cube = self.relu(cube)
        cube = self.pool2(cube)

        cube = self.conv4(cube)
        cube = self.relu(cube)

        linear=self.fc3(cube.view(b_size,-1))
        if self.target_class == 0:
            cube=linear.reshape([b_size,2,32,32,32])
        else:
            cube=linear.reshape([b_size,32,32,32])

        return cube



class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv1_out, conv3_out, conv5_out):
        super(InceptionE, self).__init__()
        self.pool_out = 3
        self.conv1_out = conv1_out
        self.conv3_out = conv3_out
        self.conv5_out = conv5_out
        self.branch1x1 = BasicConv3d(in_channels, self.conv1_out, kernel_size=1)

        self.branch3x3 = BasicConv3d(in_channels, self.conv3_out, kernel_size=3, padding = 1)
        

        self.branch5x5 = BasicConv3d(in_channels, self.conv5_out, kernel_size=5, padding = 2)
        self.branch_pool = BasicConv3d(in_channels, self.pool_out, kernel_size=1)


    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3(x)
        
        branch5x5 = self.branch5x5(x)
        
        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class Inception(nn.Module):
    def __init__(self, channels, conv1_out, conv3_out, conv5_out):
        super(Inception, self).__init__()
        self.conv1_out = conv1_out
        self.conv3_out = conv3_out
        self.conv5_out = conv5_out
        self.incep = InceptionE(channels, conv1_out = conv1_out, conv3_out = conv3_out, conv5_out = conv5_out)
        conv_in = conv1_out + conv3_out + conv5_out + 3
        self.conv1 = BasicConv3d(conv_in, conv_in, kernel_size = 3, padding = 1)
        self.conv2 = BasicConv3d(conv_in, conv_in//2, kernel_size = 3, padding = 1)
        self.conv3 = BasicConv3d(conv_in//2, 2, kernel_size = 1)
    def forward(self, x):
        b_size = x.size(0)
        incep1 = self.incep(x)
        #print('incep1 = ', incep1.size())
        conv1=self.conv1(incep1)
        conv2=self.conv2(conv1) 
        conv3=self.conv3(conv2)
        return conv3


class Inception_2(nn.Module):
    def __init__(self, channels, conv1_out, conv3_out, conv5_out, conv11_out = 25, conv12_out = 52):
        super(Inception, self).__init__()
        self.conv1_out = conv1_out
        self.conv3_out = conv3_out
        self.conv5_out = conv5_out
        self.incep = InceptionE(channels, conv1_out = conv1_out, conv3_out = conv3_out, conv5_out = conv5_out)
        conv_in = conv1_out + conv3_out + conv5_out + 3

        self.conv21 = BasicConv3d(channels, conv1_out//2, kernel_size = 3, padding = 1)
        self.conv21 = BasicConv3d(conv1_out//2, conv1_out, kernel_size = 3, padding = 1)

        self.conv21 = BasicConv3d(conv_in, conv_in, kernel_size = 3, padding = 1)
        self.conv22 = BasicConv3d(conv_in, conv_in//2, kernel_size = 3, padding = 1)
        self.conv23 = BasicConv3d(conv_in//2, 2, kernel_size = 1)
    def forward(self, x):
        b_size = x.size(0)
        conv11 = self.conv11(x)
        conv12 = self.conv12(conv11)
        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        incep1 = self.incep(conv12)
        conv21=self.conv21(incep1)
        conv22=self.conv22(conv21) 
        conv23=self.conv23(conv22)
        return conv3