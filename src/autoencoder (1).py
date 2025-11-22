import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class UNetAE(nn.Module):
    """
    Light U-Net autoencoder (RGB->RGB) with 4 down/ups and skip connections.
    """
    def __init__(self, base=32, latent=256):
        super().__init__()
        self.enc1 = conv_block(3, base)
        self.enc2 = conv_block(base, base*2)
        self.enc3 = conv_block(base*2, base*4)
        self.enc4 = conv_block(base*4, base*8)

        self.down = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base*8, latent, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent, base*8, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec4 = conv_block(base*8, base*4)

        self.up3 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec3 = conv_block(base*4, base*2)

        self.up2 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec2 = conv_block(base*2, base)

        self.out_conv = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)           # B, base, H, W
        e2 = self.enc2(self.down(e1))    # B, 2b, H/2, W/2
        e3 = self.enc3(self.down(e2))    # B, 4b, H/4, W/4
        e4 = self.enc4(self.down(e3))    # B, 8b, H/8, W/8

        b = self.bottleneck(e4)          # B, 8b, H/8, W/8

        d4 = self.up4(b)                 # B, 4b, H/4, W/4
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)                # B, 2b, H/2, W/2
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)                # B, b, H, W
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.out_conv(d2)
        out = torch.sigmoid(out)         # keep outputs in [0,1]
        return out
