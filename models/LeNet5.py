from torch import nn
from pruning_layers import SparseLinear, SparseConv2d

class ConvBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel, pool, prune):
        super().__init__()
        Conv = SparseConv2d if prune else nn.Conv2d
        self.conv = Conv(C_in,C_out,kernel)
        self.relu = nn.ReLU(inplace = True)
        self.pool = nn.MaxPool2d(2,2) if pool else nn.Identity()
        
    def forward(self, input):
        out = self.conv(input)
        out = self.relu(out)
        out = self.pool(out)
        return out
    
class LeNet5(nn.Module):
    def __init__(self, prune = False):
        super().__init__()
        linear = SparseLinear if prune else nn.Linear
        self.block1 = ConvBlock(1, 6, 5, pool = True, prune = prune)
        self.block2 = ConvBlock(6, 16, 5, pool = True, prune = prune)
        self.block3 = ConvBlock(16, 120, 5, pool = False, prune = prune)
        self.fc = nn.Sequential(
            linear(120, 84),
            nn.ReLU(inplace = True),
            linear(84, 10)
        )
        
    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(-1,120)
        out = self.fc(out)
        return out
        
