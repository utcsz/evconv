from .e2d_organization.continuous_event_datasets import ContinuousEventsDataset
from .e2d_organization.discrete_event_dataset import VoxelGridDataset

def get_continuous_dataset(path, vdataset=False):
    width, height = 346, 260
    if vdataset:
        dataset = VoxelGridDataset(base_folder=path, event_folder='events/voxels', start_time=0, stop_time=0, transform=None, normalize=True)
    else:
        dataset = ContinuousEventsDataset(base_folder=path, event_folder='events/data',\
            width=width, height=height, normalize=True, window_size = 0.05, time_shift = 0.001) # 1 ms time shift
    return dataset

def get_discrete_dataset(path):
    width, height = 346, 260
    dataset = VoxelGridDataset(base_folder=path, event_folder='events/voxels', start_time=0, stop_time=0, transform=None, normalize=True)
    return dataset





