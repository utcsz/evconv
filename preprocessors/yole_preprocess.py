import torch

def get_preprocessor():
    # return NotImplementedError('preprocess(): Not implemented yet')
    return lambda x: torch.permute(x.cuda(), [0,3,1,2])


preprocess = get_preprocessor()
