import torch.nn as nn
from .e2depth_config import config_convlstm 

from .submodules.submodules import UNetRecurrent

# ConvLSTMUNet
class ConvLSTMUNet(nn.Module):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.recurrent_block_type = str(config['recurrent_block_type'])
        self.num_bins = int(config['num_bins'])  # number of bins in the voxel grid event tensor
        self.skip_type = str(config['skip_type'])
        self.num_encoders = int(config['num_encoders'])
        self.base_num_channels = int(config['base_num_channels'])
        self.num_residual_blocks = int(config['num_residual_blocks'])
        self.norm = str(config['norm'])
        self.use_upsample_conv = bool(config['use_upsample_conv'])
        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, x):
        event_tensor, prev_states = x
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return [img_pred, states]


def get_model():
    return ConvLSTMUNet(config_convlstm)

if __name__ == '__main__':
    model = get_model()

