import torch
import torch.nn as nn
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
        self.Simple_Unet = nn.Sequential(
            self.conv_layer(self.in_channels, 16),
            self.conv_layer(16, 16),
            nn.AvgPool3d(2),
            self.conv_layer(16,32),
            nn.AvgPool3d(2),
            self.conv_layer(32,64),
            self.up_conv_layer(64, 64, 3),
            self.conv_layer(64, 32),
            self.up_conv_layer(32, 32, 3),
            self.conv_layer(32, 16),
            self.up_conv_layer(16, 16, 3),
            self.conv_layer(16, 8),
            self.up_conv_layer(8, 8, 3),
            self.conv_layer(8, 4),
            self.conv_layer(4, 1)
        )
    
    def conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        layers = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        # nn.BatchNorm3d(out_channels),
        nn.ReLU())
        return layers
    
    def up_conv_layer(self, in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, bias=True):
        layers = nn.Sequential(
        nn.ConvTranspose3d(in_channels,out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding),
#       nn.BatchNorm3d(out_channels),
        nn.ReLU())
        return layers
    
    
    def forward(self, cube):
        cube = self.Simple_Unet(cube)
        return cube