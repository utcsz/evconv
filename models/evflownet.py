import torch
import torch.nn as nn
from evflownet_config import config

class EVFlowNet(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass



def get_model():
    return EVFlowNet(config)

