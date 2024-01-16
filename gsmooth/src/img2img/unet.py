import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels,track_running_stats=False),
            nn.GroupNorm(32,mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels,track_running_stats=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self, args, img_size, param_size, bilinear=True):
        super(UNet, self).__init__()
        self.args = args
        self.img_size = img_size
        self.param_size = param_size
        self.bilinear = bilinear

        self.p_head = nn.Linear(param_size, 3*img_size**2,bias=False)
        #
        # self.p_head = nn.Sequential(
        #     nn.Linear(param_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 3 * img_size ** 2)
        # )
        self.x_head = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,3,kernel_size=3,padding=1)
        )
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)

        # self.out1 = nn.Conv2d(3,3,kernel_size=3,padding=1, bias=True)
            # nn.ReLU(),
        self.out1 = nn.Sequential(
            nn.Conv2d(6,64,kernel_size=3,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(64,3,kernel_size=3,padding=1,bias=True)
        )


    # def forward(self, x,targets, p):
    #     x = self.x_head(x)+ x + self.p_head(p).view(-1, 3, self.img_size, self.img_size)
    #
    #     x_c = torch.cat([x, targets],dim=0)
    #     x0 = x_c
    #
    #
    #     x1 = self.inc(x_c)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     x_c = self.up1(x5, x4)
    #     x_c = self.up2(x_c, x3)
    #     x_c = self.up3(x_c, x2)
    #     x_c = self.up4(x_c, x1)
    #
    #
    #     # x =  self.outc(x) +x0
    #     x_c = torch.cat([self.outc(x_c),x0],dim=1)
    #
    #     x_c = self.out1(x_c)
    #     return x_c


    def forward(self, x, p):
        x = self.x_head(x)+ x + self.p_head(p).view(-1, 3, self.img_size, self.img_size)
        x0 = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


        # x =  self.outc(x) +x0
        x = torch.cat([self.outc(x),x0],dim=1)

        x = self.out1(x)
        return x

    def sample_noise(self, x, p, noise):

        x = self.x_head(x)+ x + self.p_head(p).view(-1, 3, self.img_size, self.img_size) + noise

        x0 = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # x =  self.outc(x) +x0
        x = torch.cat([self.outc(x), x0], dim=1)

        x = self.out1(x)
        return x


    def syn_layer(self,x):
        x0 = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # x =  self.outc(x) +x0
        x = torch.cat([self.outc(x), x0], dim=1)

        x = self.out1(x)
        return x


    def fusion_layer(self,x, p):
        x = self.x_head(x) + x + self.p_head(p).view(-1, 3, self.img_size, self.img_size)
        return x

    def pre_layer(self,x):
        x = self.x_head(x) + x
        return x







# for MNIST
class MUNet(nn.Module):
    def __init__(self, args, img_size, param_size, bilinear=True):
        super(MUNet, self).__init__()
        self.args = args
        self.img_size = img_size
        self.param_size = param_size
        self.bilinear = bilinear

        self.p_head = nn.Linear(param_size, img_size**2,bias=False)
        #
        # self.p_head = nn.Sequential(
        #     nn.Linear(param_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 3 * img_size ** 2)
        # )
        self.x_head = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,1,kernel_size=3,padding=1)
        )
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

        # self.out1 = nn.Conv2d(3,3,kernel_size=3,padding=1, bias=True)
            # nn.ReLU(),
        self.out1 = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(64,1,kernel_size=3,padding=1,bias=True)
        )


    # def forward(self, x,targets, p):
    #     x = self.x_head(x)+ x + self.p_head(p).view(-1, 3, self.img_size, self.img_size)
    #
    #     x_c = torch.cat([x, targets],dim=0)
    #     x0 = x_c
    #
    #
    #     x1 = self.inc(x_c)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     x_c = self.up1(x5, x4)
    #     x_c = self.up2(x_c, x3)
    #     x_c = self.up3(x_c, x2)
    #     x_c = self.up4(x_c, x1)
    #
    #
    #     # x =  self.outc(x) +x0
    #     x_c = torch.cat([self.outc(x_c),x0],dim=1)
    #
    #     x_c = self.out1(x_c)
    #     return x_c


    def forward(self, x, p):
        x = self.x_head(x)+ x + self.p_head(p).view(-1, 1, self.img_size, self.img_size)
        x0 = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


        # x =  self.outc(x) +x0
        x = torch.cat([self.outc(x),x0],dim=1)

        x = self.out1(x)
        return x

    def sample_noise(self, x, p, noise):

        x = self.x_head(x)+ x + self.p_head(p).view(-1, 1, self.img_size, self.img_size) + noise

        x0 = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # x =  self.outc(x) +x0
        x = torch.cat([self.outc(x), x0], dim=1)

        x = self.out1(x)
        return x


    def syn_layer(self,x):
        x0 = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # x =  self.outc(x) +x0
        x = torch.cat([self.outc(x), x0], dim=1)

        x = self.out1(x)
        return x


    def fusion_layer(self,x, p):
        x = self.x_head(x) + x + self.p_head(p).view(-1, 1, self.img_size, self.img_size)
        return x

    def pre_layer(self,x):
        x = self.x_head(x) + x
        return x




