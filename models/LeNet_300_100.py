from torch import nn
from pruning_layers import SparseLinear

class LeNet(nn.Module):
    def __init__(self, prune = False):
        super().__init__()
        Linear = SparseLinear if prune else nn.Linear
        self.fc = nn.Sequential(
            Linear(784,300),
            nn.ReLU(inplace=True),
            Linear(300,100),
            nn.ReLU(inplace=True),
            Linear(100,10)
        )
        
    def forward(self, input):
        out = input.view(-1,784)
        out = self.fc(out)
        return out