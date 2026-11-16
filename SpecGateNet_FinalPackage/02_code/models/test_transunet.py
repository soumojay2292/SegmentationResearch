import torch
from transunet import TransUNet

# Instantiate model
model = TransUNet(in_ch=3, out_ch=1)

# Dummy input (batch of 2 RGB images, 128x128)
x = torch.randn(2, 3, 128, 128)

# Forward pass
y = model(x)

print("Output shape:", y.shape)
