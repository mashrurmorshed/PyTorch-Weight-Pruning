import sys

def get_model(model_type, use_gpu, prune):
    """Return selected network model.
    """
    
    if model_type == 'lenet-300-100':
        from models.LeNet_300_100 import LeNet
        net = LeNet(prune)
    else:
        print('The selected model is unavailable.')
        sys.exit()
        
        
    if use_gpu:
        net = net.cuda()
            
    return net
        