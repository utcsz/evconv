import torch
from .e2d_organization.continuous_event_datasets import ContinuousEventsDataset

def get_datset(path='/data/'):
        height, width = 180, 240
        dataset = ContinuousEventsDataset(base_folder=path, event_folder='events/data',\
                width=width, height=height, evframe_type='histogram', window_size = 0.05, time_shift = 0.001) # 1 ms time shift
        return dataset


