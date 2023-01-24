import torch

preprocess = lambda data: data["inp_voxel"].to('cuda', memory_format=torch.channels_last)

