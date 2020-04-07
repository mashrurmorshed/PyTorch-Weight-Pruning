# PyTorch Weight Pruning

An implementation of neural network pruning. In particular, this work is an attempt to implement the concepts derived in Han et al. 15, 16.

Pruning is done on fully connected layers and convolutional layers. To emulate pruning, I've defined the custom layers *SparseLinear* and *SparseConv2d*, which use bitmasks to zero out pruned connections, thereby gaining sparsity. The model directory currently contains two very easy to implement networks: the LeNet-300-100 and the LeNet5. I will gradually add more prunable models.

Defining your own custom prunable model is fairly straightforward; if you have a model definition, simply switch out nn.Conv2d layers and nn.Linear layers with their respective sparse versions. I haven't included Conv1d and Conv3d yet, as I find that they are used seldom in practice.
