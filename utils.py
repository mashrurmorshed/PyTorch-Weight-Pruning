import sys

def get_model(model_type, use_gpu, prune):
    """Return selected network model.
    """
    
    if model_type == 'lenet-300-100':
        from models.LeNet_300_100 import LeNet
        net = LeNet(prune)
    elif model_type == 'lenet5':
        from models.LeNet5 import LeNet5
        net = LeNet5(prune)
    else:
        print('The selected model is unavailable.')
        sys.exit()
        
        
    if use_gpu:
        net = net.cuda()
            
    return net
        