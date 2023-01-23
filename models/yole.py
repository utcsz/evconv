import torch
import torch.nn as nn
from .yole_config import config

class DenseObjectDet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_c = config['in_c']
        self.nr_box = config['nr_box']
        self.nr_classes = config['nr_classes']
        self.small_out_map = config['small_out_map']
        self.kernel_size = config['kernel_size']

        self.conv_layers = nn.Sequential(
            self.conv_block(in_c=self.in_c, out_c=16),
            self.conv_block(in_c=16, out_c=32),
            self.conv_block(in_c=32, out_c=64),
            self.conv_block(in_c=64, out_c=128),
            self.conv_block(in_c=128, out_c=256, max_pool=False),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.relu1 = nn.ReLU()

        if self.small_out_map:
            self.cnn_spatial_output_size = [5, 7]
        else:
            self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * 512
        self.linear_1 = nn.Linear(self.linear_input_features, 1024)
        self.linear_2 = nn.Linear(1024, spatial_size_product*(self.nr_classes + 5*self.nr_box))

    def conv_block(self, in_c, out_c, max_pool=True):
        if max_pool:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=self.kernel_size, stride=2),
                # KFencedMaskModule(0.1)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )


    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        # x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = self.relu1(x)
        x = self.linear_2(x)
        return x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])


def get_model():
    return DenseObjectDet(config)
