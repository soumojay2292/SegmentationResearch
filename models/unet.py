import torch, torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(3,64)
        self.enc2 = CBR(64,128)
        self.enc3 = CBR(128,256)
        self.enc4 = CBR(256,512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = CBR(512+256,256)
        self.dec2 = CBR(256+128,128)
        self.dec1 = CBR(128+64,64)
        self.final = nn.Conv2d(64,1,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(e4), e3],1))
        d2 = self.dec2(torch.cat([self.up(d3), e2],1))
        d1 = self.dec1(torch.cat([self.up(d2), e1],1))
        return torch.sigmoid(self.final(d1))
