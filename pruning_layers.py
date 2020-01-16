import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class SparseConv2d(nn.Module):
    """ A wrapper over Conv2D that applies a bitmask to zero out pruned connections.
    """
    def __init__(self, C_in, C_out, kernel, stride=1, padding = 0, bias = True):
        super().__init__()
        
        # Save conv arguments
        self.C_in = C_in
        self.C_out = C_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.input_shape = 0
        
        # Parameterize weight and mask for F.conv2d
        self.weight = nn.Parameter(torch.Tensor(C_out,C_in,kernel,kernel))
        self.mask = nn.Parameter(torch.ones(C_out,C_in,kernel,kernel), requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(C_out))
        else:
            self.register_parameter('bias', None) #ref https://github.com/pytorch/pytorch/issues/143
            
        self.reset_parameters()

    def reset_parameters(self):
        # https://pytorch.org/docs/stable/nn.html#conv2d  --> (Variables)
        K = 1 / np.sqrt(self.weight.shape[1]*self.weight.shape[2]*self.weight.shape[3])
        self.weight.data.uniform_(-K, K)
        if self.bias is not None:
            self.bias.data.uniform_(-K, K)

    def forward(self, input):
        self.input_shape = input.shape[2]
        return F.conv2d(input, self.weight * self.mask, self.bias, stride = self.stride, padding = self.padding)

    def prune(self, threshold):
        """ Prune connections below given threshold. 
        """
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        
        # Convert to numpy
        weight = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        # Calculate new bitmask
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply the new mask and update the existing mask
        self.weight.data = torch.from_numpy(weight * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)


class SparseLinear(nn.Module):
    """ Linear layer that applies a bitmask to zero out pruned connections.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        #Parameterize weights and mask for F.linear.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.mask = nn.Parameter(torch.ones(out_features, in_features), requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        #https://pytorch.org/docs/stable/nn.html#linear  --> (Variables)
        K = 1 / np.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-K, K)
        if self.bias is not None:
            self.bias.data.uniform_(-K, K)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def prune(self, threshold):
        """ Prune connections below given threshold.
        """
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        
        # Convert to numpy
        weight = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        # Calculate new bitmask
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply the new mask and update the existing mask
        self.weight.data = torch.from_numpy(weight * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)