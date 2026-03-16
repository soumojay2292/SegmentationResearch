import torch, torch.nn as nn
import torch.fft

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4)
        self.ff = nn.Sequential(nn.Linear(dim,dim), nn.ReLU(), nn.Linear(dim,dim))

    def forward(self,x):
        x = x.flatten(2).permute(2,0,1)
        attn,_ = self.attn(x,x,x)
        x = x+attn
        x = x+self.ff(x)
        return x.permute(1,2,0).reshape_as(x.permute(1,2,0))

class MAFFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU()
        )
        self.trans = TransformerBlock(128)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128,64,2,stride=2), nn.ReLU(),
            nn.Conv2d(64,1,1)
        )

    def forward(self,x):
        feat = self.enc(x)
        freq = torch.fft.fft2(feat)
        freq = torch.abs(freq)
        fused = feat + freq
        fused = self.trans(fused)
        return torch.sigmoid(self.dec(fused))
