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
        self.sharpening = sharpening
        self.sharp_filter = nn.Conv3d(1,1,kernel_size = 3,stride = 1, padding = 1)
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
        if self.sharpening:
            x14 = self.sharp_filter(x14)
        return x14
    