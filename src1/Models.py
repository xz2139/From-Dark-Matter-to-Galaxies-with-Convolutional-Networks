import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim

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
    
    
class Downsample(nn.Module):

    def __init__(self, in_channels, target_class):
        
        super(Downsample, self).__init__()
        self.target_class = target_class

        self.in_channels = in_channels
        self.conv1=nn.Conv3d(self.in_channels, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2=nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool1=nn.AvgPool3d(2)
        self.conv3=nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool2=nn.AvgPool3d(2)
        self.conv4=nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0, bias=True)
        self.fc3=nn.Linear(4096,32*32*32)   
        self.relu = nn.ReLU()

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
        cube = self.fc3(cube.view(b_size,-1)).squeeze(0)
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
    def __init__(self, channels, conv1_out, conv3_out, conv5_out, reg = 0):
        super(Inception, self).__init__()
        self.conv1_out = conv1_out
        self.conv3_out = conv3_out
        self.conv5_out = conv5_out
        self.incep = InceptionE(channels, conv1_out = conv1_out, conv3_out = conv3_out, conv5_out = conv5_out)
        conv_in = conv1_out + conv3_out + conv5_out + 3
        self.conv1 = BasicConv3d(conv_in, conv_in, kernel_size = 3, padding = 1)
        self.conv2 = BasicConv3d(conv_in, conv_in//2, kernel_size = 3, padding = 1)
        self.reg = reg
        if self.reg:
            dim_out = 1
        else:
            dim_out = 2
        self.conv3 = BasicConv3d(conv_in//2, dim_out, kernel_size = 1)
    def forward(self, x):
        b_size = x.size(0)
        incep1 = self.incep(x)
        #print('incep1 = ', incep1.size())
        conv1=self.conv1(incep1)
        conv2=self.conv2(conv1) 
        conv3=self.conv3(conv2)
        if self.reg:
            conv3 = conv3.squeeze(1)
        return conv3




class GridAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            #self.inter_channels = in_channels // 2
            self.inter_channels = in_channels


        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    
class Recurrent_Conv(nn.Module):
    def __init__(self, out_channels, t):
        super(Recurrent_Conv, self).__init__()
        self.t = t
        self.out_ch = out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(self.out_ch, self.out_ch, kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):
            if i==0:
                x1 = self.conv(x)    
            x1 = self.conv(x + x1)
        return x1
        
class R2CNN(nn.Module): 
    def __init__(self, in_channels, out_channels, t):
        super(R2CNN, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.t = t
        self.RCNN = nn.Sequential(
            Recurrent_Conv(self.out_ch, self.t),
            Recurrent_Conv(self.out_ch, self.t)
        )
        self.initial_conv = nn.Conv3d(self.in_ch, self.out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.initial_conv(x)
        x1 = self.RCNN(x)
        return x+x1

class UnetGating(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1)):
        super(UnetGating, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class R2Unet_atten(nn.Module):
    def __init__(self, in_channels, out_channels, t, reg = 0):
        super(R2Unet_atten, self).__init__()
        self.reg = reg
        self.in_ch= in_channels
        self.out_ch = out_channels
        self.t = t
        self.avgPool = nn.AvgPool3d(2)
        self.maxPool = nn.MaxPool3d(2)
        self.r2cnn1 = R2CNN(self.in_ch, 32, 2)
        self.r2cnn2 = R2CNN(32, 64, 2)
        self.r2cnn3 = R2CNN(64, 128, 2)
        self.r2cnn4 = R2CNN(128, 256, 2)
        self.up_conv1 = self.up_conv_layer(256, 128, 3, 2, 1, 1)     
        self.r2cnn5 = R2CNN(256, 64, 2)
        self.up_conv2 = self.up_conv_layer(64, 64, 3, 2, 1, 1)
        self.r2cnn6 = R2CNN(128, 32, 2)
        self.up_conv3 = self.up_conv_layer(32, 32, 3, 2, 1, 1)
        self.r2cnn7 = R2CNN(32, 16, 2)
    
        if self.reg == 0:
            self.conv11 = nn.Conv3d(16, 2, kernel_size = 1, stride=1, padding=0)
        else:
            self.conv11 = nn.Conv3d(16, 1, kernel_size = 1, stride=1, padding=0)
        
    
        self.gating_layer_x8 = UnetGating(128, 128)
        self.gating_layer_x10 = UnetGating(64, 64)
        
        self.attn1 = GridAttentionBlock3D(in_channels=128)
        self.attn2 = GridAttentionBlock3D(in_channels=64)
        
    def up_conv_layer(self, in_channels, out_channels, kernel_size, stride=3, padding=1, output_padding=1, bias=True):
        layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels,out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=True),
            nn.ReLU()
        )
        return layers

    def forward(self, x):
        x1 = self.r2cnn1(x)    #32
        x2 = self.maxPool(x1)  #32
        x3 = self.r2cnn2(x2)   #64
        x4 = self.maxPool(x3)  #64
        x5 = self.r2cnn3(x4)   #128
        x6 = self.maxPool(x5)  #128
        x7 = self.r2cnn4(x6)   #256
        x8 = self.up_conv1(x7) #128
        #new
        gating_x8 = self.gating_layer_x8(x8) #128 
        #print(' gating_x8.size= ', gating_x8.size())
        x5_atted, _ = self.attn1(x5, gating_x8) #128
        x8 = torch.cat((x5_atted, x8), dim = 1) #256
        x9 = self.r2cnn5(x8)  #64
        x10 = self.up_conv2(x9) #64
        #new
        gating_x10 = self.gating_layer_x10(x10) #64
        x3_atted, _ = self.attn2(x3, gating_x10) #64
        x10 = torch.cat((x3_atted, x10), dim = 1) #128
        x11 = self.r2cnn6(x10) #32
        x12 = self.up_conv3(x11)
        #x12 = torch.cat((x1, x12), dim = 1)
        x13 = self.r2cnn7(x12)
        x14 = self.conv11(x13)
        if self.reg:
            x14 = x14.squeeze(1)
        return x14

class R2Unet(nn.Module):
    def __init__(self, in_channels, out_channels, t, reg = 0, sharpening = False):
        super(R2Unet, self).__init__()
        self.reg = reg
        self.in_ch= in_channels
        self.out_ch = out_channels
        self.t = t
        self.avgPool = nn.AvgPool3d(2)
        self.maxPool = nn.MaxPool3d(2)
        self.r2cnn1 = R2CNN(self.in_ch, 32, 2)
        self.r2cnn2 = R2CNN(32, 64, 2)
        self.r2cnn3 = R2CNN(64, 128, 2)
        self.r2cnn4 = R2CNN(128, 256, 2)
        self.up_conv1 = self.up_conv_layer(256, 128, 3, 2, 1, 1)     
        self.r2cnn5 = R2CNN(256, 64, 2)
        self.up_conv2 = self.up_conv_layer(64, 64, 3, 2, 1, 1)
        self.r2cnn6 = R2CNN(128, 32, 2)
        self.up_conv3 = self.up_conv_layer(32, 32, 3, 2, 1, 1)
        self.r2cnn7 = R2CNN(32, 16, 2)
        if self.reg == 0:
            self.conv11 = nn.Conv3d(16, 2, kernel_size = 1, stride=1, padding=0)
        else:
            self.conv11 = nn.Conv3d(16, 1, kernel_size = 1, stride=1, padding=0)
    def up_conv_layer(self, in_channels, out_channels, kernel_size, stride=3, padding=1, output_padding=1, bias=True):
        layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels,out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=True),
            nn.ReLU()
        )
        return layers
    
    def forward(self, x):
        x1 = self.r2cnn1(x)    
        x2 = self.maxPool(x1)  
        x3 = self.r2cnn2(x2)   
        x4 = self.maxPool(x3)  
        x5 = self.r2cnn3(x4)   
        x6 = self.maxPool(x5)  
        x7 = self.r2cnn4(x6)   
        x8 = self.up_conv1(x7) 
        x8 = torch.cat((x5, x8), dim = 1)
        x9 = self.r2cnn5(x8)
        x10 = self.up_conv2(x9)
        x10 = torch.cat((x3, x10), dim = 1)
        x11 = self.r2cnn6(x10)
        x12 = self.up_conv3(x11)
        #x12 = torch.cat((x1, x12), dim = 1)
        x13 = self.r2cnn7(x12)
        x14 = self.conv11(x13)
        if self.reg:
            x14 = x14.squeeze(1)
        return x14


class R2Unet_subhalo(nn.Module):
    def __init__(self, in_channels, out_channels, t, reg = 0, sharpening = False):
        super(R2Unet_subhalo, self).__init__()
        self.reg = reg
        self.in_ch= in_channels
        self.out_ch = out_channels
        self.t = t
        self.avgPool = nn.AvgPool3d(2)
        self.maxPool = nn.MaxPool3d(2)
        self.r2cnn1 = R2CNN(self.in_ch, 32, 2)
        self.r2cnn2 = R2CNN(32, 64, 2)
        self.r2cnn3 = R2CNN(64, 128, 2)
        self.r2cnn4 = R2CNN(256, 256, 2)
        self.up_conv1 = self.up_conv_layer(256, 128, 3, 2, 1, 1)     
        self.r2cnn5 = R2CNN(256, 64, 2)
        self.up_conv2 = self.up_conv_layer(64, 128, 3, 2, 1, 1)
        self.r2cnn6 = R2CNN(128, 32, 2)
        self.up_conv3 = self.up_conv_layer(32, 32, 3, 2, 1, 1)
        self.r2cnn7 = R2CNN(32, 16, 2)
        if self.reg == 0:
            self.conv11 = nn.Conv3d(16, 2, kernel_size = 1, stride=1, padding=0)
        else:
            self.conv11 = nn.Conv3d(16, 1, kernel_size = 1, stride=1, padding=0)
    def up_conv_layer(self, in_channels, out_channels, kernel_size, stride=3, padding=1, output_padding=1, bias=True):
        layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels,out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=True),
            nn.ReLU()
        )
        return layers
    
    
    def forward(self, subhalo_mass, dark_count):
        '''
        subhalo count
        '''
        x1_s = self.r2cnn1(subhalo_mass)
        x2_s = self.maxPool(x1_s)  
        x3_s = self.r2cnn2(x2_s)   
        x4_s = self.maxPool(x3_s)  
        x5_s = self.r2cnn3(x4_s)   
        x6_s = self.maxPool(x5_s)  
        '''
        dark matter count 
        '''
        x1_d = self.r2cnn1(dark_count)    
        x2_d = self.maxPool(x1_d)  
        x3_d = self.r2cnn2(x2_d)   
        x4_d = self.maxPool(x3_d)  
        x5_d = self.r2cnn3(x4_d)   
        x6_d = self.maxPool(x5_d)  
        
        x_6 = torch.cat((x6_s, x6_d), dim = 1)
        x7 = self.r2cnn4(x_6)
        x8 = self.up_conv1(x7)
        
        x8 = torch.cat((x5_s, x8), dim = 1)
        x9 = self.r2cnn5(x8)
        x10 = self.up_conv2(x9)
        #x10 = torch.cat((x3_s, x10), dim = 1)
        x11 = self.r2cnn6(x10)
        x12 = self.up_conv3(x11)
        #x12 = torch.cat((x1, x12), dim = 1)
        x13 = self.r2cnn7(x12)
        x14 = self.conv11(x13)
        if self.reg:
            x14 = x14.squeeze(1)
        return x14

class one_layer_conv(nn.Module):
    def __init__(self,in_channels, one_layer_outchannel, kernel_size, non_linearity, transformation, power):
        super(one_layer_conv,self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels,one_layer_outchannel,kernel_size, padding = 1),
            getattr(nn,non_linearity)(), 
            nn.Conv3d(one_layer_outchannel,1,1), 
            nn.ReLU()
            )
        for m in self.model:
            if isinstance(m,nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)        
        self.transformation = transformation
        self.power = power

    def forward(self,X):
        if self.transformation == 'sqrt_root':
            X = X.pow(self.power)
        elif self.transformation == 'log':
            X[X == 0] = 1
            X = torch.log(X)
        elif self.transformation == 'default':
            X = X
        else:
            raise ValueError('Wrong data preprocessing procedure!')
            X = X/10
        return self.model(X)

    
class two_phase_conv(nn.Module):
    def __init__(self,first_pmodel,second_pmodel, thres =0.5):
        super(two_phase_conv,self).__init__()
        self.fp = first_pmodel
        for param in self.fp.parameters():
            param.requires_grad = False
        self.sp = second_pmodel
        self.thres = thres
    
    def forward(self,X):
        output = self.fp(X)
        outputs = F.softmax(output, dim=1)[:,1,:,:,:]
        #print(' outputs.size= ', outputs.size())
        mask_value = (outputs > self.thres).float()
        #print('mask_value.size= ', mask_value.size())
        #print(' self.sp(X).size= ', self.sp(X).size())
        result = mask_value * self.sp(X)
        return result    
    

#class two_phase_conv(nn.Module):
#    def __init__(self,first_pmodel,second_pmodel, thres =0.5):
#        super(two_phase_conv,self).__init__()
#        self.fp = first_pmodel
#        for param in self.fp.parameters():
#            param.requires_grad = False
#        self.sp = second_pmodel
#        self.thres = thres
    
#    def forward(self, X, subhalo_mass):

#        mask_value = (subhalo_mass[:,0,:,:,:] > 0).float()
 
#        result = mask_value * self.sp(subhalo_mass, X)
#        return result
