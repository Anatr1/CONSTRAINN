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
    This version accepts separate decoder (x1) and skip connection (x2) channel sizes.
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
        # Pad if necessary to match dimensions with x2
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
        Medium U-Net:
          Encoder:
            inc:       n_channels  -> 64
            down1:     64 -> 128
            down2:     128 -> 256
            down3:     256 -> 512
            down4:     512 -> 1024
          Decoder:
            up1:       1024 + 512 -> 512
            up2:       512 + 256 -> 256
            up3:       256 + 128 -> 128
            up4:       128 + 64 -> 64
          Output: 1x1 conv (64 -> n_classes)
        """
        super(UNet, self).__init__()
        self.inc    = DoubleConv(n_channels, 64)
        self.down1  = Down(64, 128)
        self.down2  = Down(128, 256)
        self.down3  = Down(256, 512)
        self.down4  = Down(512, 1024)
        
        # Optional dropout in bottleneck
        self.bottleneck_dropout = nn.Dropout2d(0.3)
        
        self.up1 = Up(in_channels=1024, skip_channels=512, out_channels=512, bilinear=bilinear)
        self.up2 = Up(in_channels=512, skip_channels=256, out_channels=256, bilinear=bilinear)
        self.up3 = Up(in_channels=256, skip_channels=128, out_channels=128, bilinear=bilinear)
        self.up4 = Up(in_channels=128, skip_channels=64, out_channels=64, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)
        self.final_activation = final_activation
        
    def forward(self, x):
        x1 = self.inc(x)      # (batch, 64, H, W)
        x2 = self.down1(x1)   # (batch, 128, H/2, W/2)
        x3 = self.down2(x2)   # (batch, 256, H/4, W/4)
        x4 = self.down3(x3)   # (batch, 512, H/8, W/8)
        x5 = self.down4(x4)   # (batch, 1024, H/16, W/16)
        x5 = self.bottleneck_dropout(x5)
        x = self.up1(x5, x4)  # (batch, 512, H/8, W/8)
        x = self.up2(x, x3)   # (batch, 256, H/4, W/4)
        x = self.up3(x, x2)   # (batch, 128, H/2, W/2)
        x = self.up4(x, x1)   # (batch, 64, H, W)
        logits = self.outc(x) # (batch, n_classes, H, W)
        
        if self.final_activation == 'sigmoid':
            return torch.sigmoid(logits)
        elif self.final_activation == 'softmax':
            return torch.softmax(logits, dim=1)
        else:
            return logits