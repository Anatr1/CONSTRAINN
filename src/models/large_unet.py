import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Padding if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, final_activation=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc   = DoubleConv(n_channels, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 1024)
        self.down4 = Down(1024, 2048)
        self.down5 = Down(2048, 4096)

        # Optional dropout in the bottleneck
        self.bottleneck_dropout = nn.Dropout2d(0.3)

        # Decoder with updated in_channels for concatenation
        self.up1 = Up(4096 + 2048, 2048, bilinear)  # 6144 -> 2048
        self.up2 = Up(2048 + 1024, 1024, bilinear)    # 3072 -> 1024
        self.up3 = Up(1024 + 512, 512, bilinear)      # 1536 -> 512
        self.up4 = Up(512 + 256, 256, bilinear)       # 768 -> 256
        self.up5 = Up(256 + 128, 128, bilinear)       # 384 -> 128

        self.outc = OutConv(128, n_classes)
        self.final_activation = final_activation

    def forward(self, x):
        x1 = self.inc(x)      # 128 channels
        x2 = self.down1(x1)   # 256 channels
        x3 = self.down2(x2)   # 512 channels
        x4 = self.down3(x3)   # 1024 channels
        x5 = self.down4(x4)   # 2048 channels
        x6 = self.down5(x5)   # 4096 channels
        x6 = self.bottleneck_dropout(x6)
        x = self.up1(x6, x5)  # up1 expects 4096 + 2048 = 6144 channels, outputs 2048
        x = self.up2(x, x4)   # 2048 + 1024 = 3072 channels, outputs 1024
        x = self.up3(x, x3)   # 1024 + 512 = 1536 channels, outputs 512
        x = self.up4(x, x2)   # 512 + 256 = 768 channels, outputs 256
        x = self.up5(x, x1)   # 256 + 128 = 384 channels, outputs 128
        logits = self.outc(x)
        
        if self.final_activation == 'sigmoid':
            return torch.sigmoid(logits)
        elif self.final_activation == 'softmax':
            return torch.softmax(logits, dim=1)
        else:
            return logits