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
    """
    Upscaling then double conv.
    The constructor takes in the number of channels from the upsampled input (x1)
    and from the skip connection (x2), so that after concatenation the DoubleConv
    will receive (in_channels + skip_channels) as input.
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if necessary
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1x1 convolution to produce final output."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, final_activation=None):
        """
        :param n_channels: Number of input channels.
        :param n_classes: Number of output channels.
        :param bilinear: Whether to use bilinear upsampling.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder: using a moderate number of filters.
        self.inc = DoubleConv(n_channels, 32)       # output: 32 channels
        self.down1 = Down(32, 64)                    # output: 64 channels
        self.down2 = Down(64, 128)                   # output: 128 channels
        self.down3 = Down(128, 256)                  # output: 256 channels
        
        # If bilinear upsampling, we do not use ConvTranspose2d.
        self.down4 = Down(256, 512)                  # output: 512 channels
        
        # Bottleneck dropout
        self.bottleneck_dropout = nn.Dropout2d(0.2)
        
        # Decoder:
        self.up1 = Up(in_channels=512, skip_channels=256, out_channels=256, bilinear=bilinear)
        self.up2 = Up(in_channels=256, skip_channels=128, out_channels=128, bilinear=bilinear)
        self.up3 = Up(in_channels=128, skip_channels=64, out_channels=64, bilinear=bilinear)
        self.up4 = Up(in_channels=64, skip_channels=32, out_channels=32, bilinear=bilinear)
        self.outc = OutConv(32, n_classes)
        self.final_activation = final_activation
        
    def forward(self, x):
        x1 = self.inc(x)      # shape: (batch, 32, H, W)
        x2 = self.down1(x1)   # shape: (batch, 64, H/2, W/2)
        x3 = self.down2(x2)   # shape: (batch, 128, H/4, W/4)
        x4 = self.down3(x3)   # shape: (batch, 256, H/8, W/8)
        x5 = self.down4(x4)   # shape: (batch, 512, H/16, W/16)
        x5 = self.bottleneck_dropout(x5)
        x = self.up1(x5, x4)  # up1: receives (512 upsampled) and skip 256
        x = self.up2(x, x3)   # up2: (decoder channels 256, skip 128)
        x = self.up3(x, x2)   # up3: (decoder channels 128, skip 64)
        x = self.up4(x, x1)   # up4: (decoder channels 64, skip 32) --> outputs 32 channels
        logits = self.outc(x)
        
        if self.final_activation == 'sigmoid':
            return torch.sigmoid(logits)
        elif self.final_activation == 'softmax':
            return torch.softmax(logits, dim=1)
        else:
            return logits