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
        '''
        torch.nn.Conv3d: input(N,C,D,H,W)
                            output(N,C,Dout,Hout,Wout) 
        torch.nn.AvgPool3d: input(N,C,D,H,W)
                            output(N,C,Dout,Hout,Wout)     
        '''
        '''
        nn.Conv3d(in_channels, out_channels, kernel_size)
        nn.AvgPool3d()
        '''
        self.draft_model = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),
            #nn.BatchNorm3d(out_ch),
#             nn.AvgPool3d(3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
#             nn.AvgPool3d(3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, cube):
        cube = self.draft_model(cube)
        return cube
    
class SimpleUnet(nn.Module):

    def __init__(self, in_channels):
        
        super(SimpleUnet, self).__init__()
        self.in_channels = in_channels
        self.conv1=nn.Conv3d(self.in_channels, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2=nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool1=nn.AvgPool3d(2)
        self.conv3=nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool2=nn.AvgPool3d(2)
        self.conv4=nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0, bias=True)
#         self.upsamp1= nn.Upsample(scale_factor=2, mode='nearest')
#         self.upconv1=nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)        
#         self.conv5=nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=0, bias=True)
#         self.upsamp2= nn.Upsample(scale_factor=2, mode='nearest')
#         self.upconv2=nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv6=nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=0, bias=True)
#         self.upsamp3= nn.Upsample(scale_factor=2, mode='nearest')
#         self.upconv3=nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv7=nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=0, bias=True)
#         self.upsamp4= nn.Upsample(scale_factor=2, mode='nearest')
#         self.upconv4=nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv8=nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=0, bias=True)
#         self.conv9=nn.Conv3d(4, 1, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu=nn.ReLU()
#         self.fc1=nn.Linear(43904,32768)
#         self.fc2=nn.Linear(6912, 32768)
        self.fc3=nn.Linear(4096,32768*2)
#         self.Simple_Unet = nn.Sequential(
#             self.conv_layer(self.in_channels, 16),
#             self.conv_layer(16, 16),
#             nn.AvgPool3d(2),
#             self.conv_layer(16,32),
#             nn.AvgPool3d(2),
#             self.conv_layer(32,64),
#             self.up_conv_layer(64, 64, 3),
#             self.conv_layer(64, 32),
#             self.up_conv_layer(32, 32, 3),
#             self.conv_layer(32, 16),
#             self.up_conv_layer(16, 16, 3),
#             self.conv_layer(16, 8),
#             self.up_conv_layer(8, 8, 3),
#             self.conv_layer(8, 4),
#             self.conv_layer(4, 1)
#         )
    
#     def conv_layer(self, inputs, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
# #         layers = nn.Sequential(
#         nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
#         # nn.BatchNorm3d(out_channels),
#         nn.ReLU()
# #         print(layers.size())
#         return layers
    
#     def up_conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
#         layers = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             # should be feat_in*2 or feat_in
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             # nn.BatchNorm3d(out_channels),
#             nn.ReLU())
#         return layers
    
    
    def forward(self, cube):
        b_size=cube.size()[0]
        cube = self.conv1(cube)
        cube = self.relu(cube)
#         print(cube.size())
        cube = self.conv2(cube)
        cube = self.relu(cube)
#         print(cube.size())
        cube = self.pool1(cube)
#         print(cube.size())
#         linear1=self.fc1(cube.view(b_size,-1))
        cube = self.conv3(cube)
        cube = self.relu(cube)
#         print(cube.size())
        cube = self.pool2(cube)
#         print(cube.size())
#         linear2=self.fc2(cube.view(b_size,-1))
        cube = self.conv4(cube)
        cube = self.relu(cube)
#         print(cube.size())
        linear=self.fc3(cube.view(b_size,-1))
#         linear=linear1+linear2+linear
        #print('linear: ', linear.size())
        cube=linear.reshape([b_size,2,32,32,32])
#         cube = self.relu(cube)
#         cube = self.upsamp1(cube)
#         cube = self.upconv1(cube)
#         cube = self.relu(cube)
#         cube = self.conv5(cube)
#         cube = self.relu(cube)
#         cube = self.upsamp2(cube)
#         cube = self.upconv2(cube)
#         cube = self.relu(cube)
#         cube = self.conv6(cube)
#         cube = self.relu(cube)
#         cube = self.upsamp3(cube)
#         cube = self.upconv3(cube)
#         cube = self.relu(cube)        
#         cube = self.conv7(cube)
#         cube = self.relu(cube)
#         cube = self.upsamp4(cube)
#         cube = self.upconv4(cube)
#         cube = self.relu(cube) 
#         cube = self.conv8(cube)
#         cube = self.relu(cube)
#         cube = self.conv9(cube)
#         cube = self.relu(cube)
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
    def __init__(self, channels, conv1_out = 30, conv3_out = 40, conv5_out = 50):
        super(Inception, self).__init__()
        self.conv1_out = conv1_out
        self.conv3_out = conv3_out
        self.conv5_out = conv5_out
        self.incep = InceptionE(channels, conv1_out = conv1_out, conv3_out = conv3_out, conv5_out = conv5_out)
        conv_in = conv1_out + conv3_out + conv5_out + 3
        self.conv1 = BasicConv3d(conv_in, 123, kernel_size = 5, padding = 2)
        self.conv2 = BasicConv3d(123, 60, kernel_size = 3, padding = 1)
        self.conv3 = BasicConv3d(60, 2, kernel_size = 1)
    def forward(self, x):
        b_size = x.size(0)
        incep1 = self.incep(x)
        #print('incep1 = ', incep1.size())
        conv1=self.conv1(incep1)
        conv2=self.conv2(conv1) 
        conv3=self.conv3(conv2)
        return conv3
