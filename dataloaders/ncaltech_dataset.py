import torch
from .ncaltech_organization import NCaltech101
from .ncaltech_organization import NCars

# from .e2d_organization.continuous_event_datasets import ContinuousEventsDataset

# def get_dataset(path='/data/'):
#         height, width = 180, 240
#         dataset = ContinuousEventsDataset(base_folder=path, event_folder='events/data',\
#                 width=width, height=height, evframe_type='histogram', window_size = 0.05, time_shift = 0.001) # 1 ms time shift
#         return dataset


def get_caltech_dataset(path='/data/'):
        height, width = 180, 240
        dataset = NCaltech101(root=path, object_classes='all', height=height, width=width, nr_events_window=1000, augmentation=None, mode='validation', event_representation='histogram') # 1 ms time shift
        return dataset



