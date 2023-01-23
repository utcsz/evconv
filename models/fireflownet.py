import torch
import torch.nn as nn
from .submodules.submodules import ConvLayer, ResidualBlock
from .fireflownet_config import config as fireflownet_config

class FireFlowNet(nn.Module):
    """
    FireFlowNet architecture for (dense/sparse) optical flow estimation from event-data.
    "Back to Event Basics: Self Supervised Learning of Image Reconstruction from Event Data via Photometric Constancy", Paredes-Valles et al., 2020
    """

    def __init__(self, config):
        super().__init__()
        base_num_channels = config["base_num_channels"]
        kernel_size = config["kernel_size"]
        num_bins = config["num_bins"]
        # self.mask = unet_kwargs["mask_output"]

        padding = kernel_size // 2
        self.E1 = ConvLayer(num_bins, base_num_channels, kernel_size, padding=padding)
        self.E2 = ConvLayer(base_num_channels, base_num_channels, kernel_size, padding=padding)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.E3 = ConvLayer(base_num_channels, base_num_channels, kernel_size, padding=padding)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=2, kernel_size=1, activation="tanh")

    def forward(self, x_incr):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        inp_voxel = x_incr

        # forward pass
        x_incr = inp_voxel
        x_incr = self.E1(x_incr)
        x_incr = self.E2(x_incr)
        x_incr = self.R1(x_incr)
        x_incr = self.E3(x_incr)
        x_incr = self.R2(x_incr)
        flow = self.pred(x_incr)

        # # mask flow
        # if self.mask:
        #     mask = torch.sum(inp_cnt, dim=1, keepdim=True)
        #     mask[mask > 0] = 1
        #     flow = flow * mask

        return flow




def get_model():
    return FireFlowNet(fireflownet_config)


if __name__ == '__main__':
    model = get_model()


