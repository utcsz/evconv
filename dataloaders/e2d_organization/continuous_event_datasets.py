import random
from os.path import join
from .base_event_dataset import EventDataset
from .util import event_histogram, first_element_greater_than, last_element_less_than, normalize_voxelgrid, events_to_voxel_grid_absT
import numpy as np


class RawEventsDataset(EventDataset):

    def parse_event_folder(self):
        self.num_bins = None

    def num_channels(self):
        return self.num_bins

    def __getitem__(self, i, transform_seed=None):
        assert(i >= 0)
        assert(i < self.length)
        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)
        events_raw = np.load(join(self.event_folder, 'events_{:010d}.npy'.format(self.first_valid_idx + i)))
        return events_raw


class ContinuousEventsRawDataset(EventDataset):

    def __init__(self, base_folder, event_folder, start_time=0, stop_time=0, transform=None, normalize=True, window_size = 0.05, time_shift=0.001, delta=0) :
        super().__init__(base_folder, event_folder, start_time=start_time, stop_time=stop_time, transform=transform, normalize=normalize)

        self.window_size = window_size
        self.time_shift = time_shift
        self.delta = delta

        # self.first_valid_idx, self.last_valid_idx is actually the first/last valid file.

        if self.start_time <= 0.0:
            self.lowerlimit_idx = 0
        else:
            min_stamp_idx, val = first_element_greater_than(self.stamps, self.start_time)
            assert(val is not None)
            self.lowerlimit_idx = int( (val - self.stamps[0])/time_shift)

        if self.stop_time <= 0.0:
            self.upperlimit_idx = int((self.stamps[-1]- window_size - delta - self.stamps[0]) / time_shift)
        else:
            max_stamp_idx, val = last_element_less_than(self.stamps, self.stop_time)
            assert(val is not None)
            self.upperlimit_idx = int( (val - window_size - delta - self.stamps[0])/time_shift )

    def parse_event_folder(self):
        self.num_bins = None

    def __len__(self):
        return self.upperlimit_idx - self.lowerlimit_idx + 1

    def __getitem__(self, i):

        timestamp_start = (self.lowerlimit_idx + i)*self.time_shift + self.delta
        timestamp_end = timestamp_start + self.window_size

        first_file_idx = int(max(first_element_greater_than(self.stamps, timestamp_start)[0] - 1, self.first_valid_idx))
        last_file_idx = int(min(last_element_less_than(self.stamps, timestamp_end)[0]+1, self.last_valid_idx))

        events_raw = []
        for file_idx in range(first_file_idx, last_file_idx+1):
            events_raw.append( np.load(join(self.event_folder, 'events_{:010d}.npy'.format(file_idx))) )

        events_raw = np.array(np.concatenate(events_raw))
        # self.initial_stamp = 0.001
        # print(events_raw[:, 0]/1e9, self.initial_stamp, self.initial_stamp + timestamp_start)
        idx_start = first_element_greater_than(events_raw[:,0]/1e9, self.initial_stamp + timestamp_start)[0]
        idx_end = last_element_less_than(events_raw[:,0]/1e9, self.initial_stamp + timestamp_end)[0]

#        print(idx_start, idx_end)
        return events_raw[idx_start:idx_end], self.initial_stamp + timestamp_start

class ContinuousEventsDataset(ContinuousEventsRawDataset):

    def __init__(self, base_folder, event_folder, width, height, start_time=0, stop_time=0, transform=None, normalize=False, evframe_type='voxelgrid', window_size = 0.05, time_shift=0.001, num_bins=5, delta=0.):
        super().__init__(base_folder, event_folder, start_time, stop_time, transform, normalize, window_size, time_shift, delta=delta)
        self.width = width
        self.height = height
        self.num_bins = num_bins
        self.evframe_type = evframe_type
    
    def __getitem__(self, i):
        events_raw, startT = super().__getitem__(i)
        if self.evframe_type == 'histogram':
            event_voxel_grid = event_histogram(events_raw, W=self.width, H=self.height)
        elif self.evframe_type == 'voxelgrid':
            event_voxel_grid = events_to_voxel_grid_absT(events_raw, num_bins=self.num_bins, width=self.width, height=self.height, deltaT=self.window_size, startT=startT)
        else:
            raise NotImplementedError

        if self.normalize:
            event_voxel_grid = normalize_voxelgrid(event_voxel_grid)
        if self.transform:
            event_voxel_grid = self.transform(event_voxel_grid)

        return event_voxel_grid


# if __name__ == '__main__':
    # import cv2
    # base_folder = '/home/sankeerth/ev/depth_proj/data/test/'
    # dataset4 = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data/', width=346, height=260, window_size=0.002, time_shift=0.0005)
    # dataloader4 = DataLoader(dataset4, shuffle=False, batch_size=1)
    # for i in dataloader4:
    #     img = torch.sum(i[0], dim=0).cpu().numpy()
    #     img = img - np.min(img) / (np.max(img) - np.min(img))
    #     img = cv2.Mat(img)
    #     cv2.imshow('a', img)
    #     cv2.waitKey(10)

    # dataset3 = ShiftedRawEventsDataset(base_folder='./data/test/', event_folder='events/data/')
    # dataloader3 = DataLoader(dataset3)
    # d3, d3sh = iter(dataloader3).next()
    # print(d3, d3sh)


