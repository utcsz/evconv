import os
import tqdm
import random
import numpy as np
from os import listdir
from os.path import join
# import event_representations as er
from numpy.lib import recfunctions as rfn


def random_shift_events(events, max_shift=20, resolution=(180, 240), bounding_box=None):
    H, W = resolution
    if bounding_box is not None:
        x_shift = np.random.randint(-min(bounding_box[0, 0], max_shift),
                                    min(W - bounding_box[2, 0], max_shift), size=(1,))
        y_shift = np.random.randint(-min(bounding_box[0, 1], max_shift),
                                    min(H - bounding_box[2, 1], max_shift), size=(1,))
        bounding_box[:, 0] += x_shift
        bounding_box[:, 1] += y_shift
    else:
        x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))

    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    if bounding_box is None:
        return events

    return events, bounding_box


def random_flip_events_along_x(events, resolution=(180, 240), p=0.5, bounding_box=None):
    H, W = resolution
    flipped = False
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
        flipped = True

    if bounding_box is None:
        return events

    if flipped:
        bounding_box[:, 0] = W - 1 - bounding_box[:, 0]
        bounding_box = bounding_box[[1, 0, 3, 2]]
    return events, bounding_box



class NCaltech101:
    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='training',
                 event_representation='histogram', shuffle=True):
        """
        Creates an iterator over the N_Caltech101 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        root = os.path.join(root, mode)

        if object_classes == 'all':
            self.object_classes = listdir(root)
        else:
            self.object_classes = object_classes

        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.event_representation = event_representation

        self.files = []
        self.labels = []

        for i, object_class in enumerate(self.object_classes):
            new_files = [join(root, object_class, f) for f in listdir(join(root, object_class))]
            self.files += new_files
            self.labels += [i] * len(new_files)

        self.nr_samples = len(self.labels)

        if shuffle:
            zipped_lists = list(zip(self.files, self.labels))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files, self.labels = zip(*zipped_lists)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        filename = self.files[idx]
        events = np.load(filename).astype(np.float32)
        nr_events = events.shape[0]

        window_start = 0
        window_end = nr_events
        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)
            window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))

        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end = min(nr_events, window_start + self.nr_events_window)

        events = events[window_start:window_end, :]

        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, label, histogram

    def generate_input_representation(self, events, shape):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        if self.event_representation == 'histogram':
            return self.generate_event_histogram(events, shape)
        elif self.event_representation == 'event_queue':
            return self.generate_event_queue(events, shape)

    @staticmethod
    def generate_event_histogram(events, shape):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events.T
        x = x.astype(np.int)
        y = y.astype(np.int)

        img_pos = np.zeros((H * W,), dtype="float32")
        img_neg = np.zeros((H * W,), dtype="float32")

        np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
        np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))

        return histogram

    @staticmethod
    def generate_event_queue(events, shape, K=15):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        events = events.astype(np.float32)

        if events.shape[0] == 0:
            return np.zeros([H, W, 2*K], dtype=np.float32)

        # [2, K, height, width],  [0, ...] time, [:, 0, :, :] newest events
        four_d_tensor = er.event_queue_tensor(events, K, H, W, -1).astype(np.float32)

        # Normalize
        four_d_tensor[0, ...] = four_d_tensor[0, 0, None, :, :] - four_d_tensor[0, :, :, :]
        max_timestep = np.amax(four_d_tensor[0, :, :, :], axis=0, keepdims=True)

        # four_d_tensor[0, ...] = np.divide(four_d_tensor[0, ...], max_timestep, where=max_timestep.astype(np.bool))
        four_d_tensor[0, ...] = four_d_tensor[0, ...] / (max_timestep + (max_timestep == 0).astype(np.float))

        return four_d_tensor.reshape([2*K, H, W]).transpose(1, 2, 0)


class NCaltech101_ObjectDetection(NCaltech101):
    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='training',
                 event_representation='histogram', shuffle=True):
        """
        Creates an iterator over the N_Caltech101 object recognition dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        if object_classes == 'all':
            self.object_classes = listdir(os.path.join(root, 'Caltech101'))
        else:
            self.object_classes = object_classes
        self.object_classes.sort()

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.event_representation = event_representation

        self.files = []
        self.class_labels = []
        self.bounding_box_list = []

        self.createDataset()
        self.nr_samples = len(self.files)

        if shuffle:
            zipped_lists = list(zip(self.files,  self.class_labels,  self.bounding_box_list))
            random.shuffle(zipped_lists)
            self.files,  self.class_labels,  self.bounding_box_list = zip(*zipped_lists)

    def createDataset(self):
        """Does a stratified training, testing, validation split"""
        np_random_state = np.random.RandomState(42)
        training_ratio = 0.7
        testing_ratio = 0.2
        # Validation Ratio will be 0.1

        for i_class, object_class in enumerate(self.object_classes):
            if object_class == 'BACKGROUND_Google':
                continue
            dir_path = os.path.join(self.root, 'Caltech101', object_class)
            image_files = listdir(dir_path)
            nr_samples = len(image_files)

            random_permutation = np_random_state.permutation(nr_samples)
            nr_samples_train = int(nr_samples*training_ratio)
            nr_samples_test = int(nr_samples*testing_ratio)

            if self.mode == 'training':
                start_idx = 0
                end_idx = nr_samples_train
            elif self.mode == 'testing':
                start_idx = nr_samples_train
                end_idx = nr_samples_train + nr_samples_test
            elif self.mode == 'validation':
                start_idx = nr_samples_train + nr_samples_test
                end_idx = nr_samples

            for idx in random_permutation[start_idx:end_idx]:
                self.files.append(os.path.join(self.root, 'Caltech101', object_class, image_files[idx]))
                annotation_file = 'annotation' + image_files[idx][5:]
                self.readBoundingBox(os.path.join(self.root, 'Caltech101_annotations', object_class, annotation_file))
                self.class_labels.append(i_class)

    def readBoundingBox(self, file_path):
        f = open(file_path)
        annotations = np.fromfile(f, dtype=np.int16)
        f.close()
        self.bounding_box_list.append(annotations[2:10])

    def loadEventsFile(self, file_name):
        f = open(file_name, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        all_p = all_p.astype(np.float64)
        all_p[all_p == 0] = -1

        return np.column_stack((all_x, all_y, all_ts, all_p))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        class_label = self.class_labels[idx]
        bounding_box = self.bounding_box_list[idx]
        bounding_box = bounding_box.reshape([4, 2])
        # Set negative corners to zero
        bounding_box = np.maximum(bounding_box, np.zeros_like(bounding_box))
        filename = self.files[idx]
        events = self.loadEventsFile(filename).astype(np.float32)
        nr_events = events.shape[0]

        window_start = 0
        window_end = nr_events
        if self.augmentation:
            window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))
        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end = min(nr_events, window_start + self.nr_events_window)
        events = events[window_start:window_end, :]

        bounding_box = self.moveBoundingBox(bounding_box, events[-1, 2])

        if self.augmentation:
            events, bounding_box = random_flip_events_along_x(events, resolution=(self.height, self.width),
                                                              bounding_box=bounding_box)
            events, bounding_box = random_shift_events(events, resolution=(self.height, self.width),
                                                       bounding_box=bounding_box)

        # Required Format: ['x', 'y', 'w', 'h', 'class_id'].  (x, y) is top left point\
        new_format_bbox = np.concatenate([bounding_box[0, :], bounding_box[2, :] - bounding_box[0, :],
                                          np.array([class_label])])

        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, new_format_bbox[np.newaxis, :], histogram

    def moveBoundingBox(self, bounding_box, current_time):
        """
        Move bounding box according the motion of the event camera. Code was adopted from the matlab code provided by
        the dataset authors.
        """
        current_time = float(current_time)
        if current_time < 100e3:
            bounding_box[:, 0] = bounding_box[:, 0] + 3.5 * current_time / 100e3
            bounding_box[:, 1] = bounding_box[:, 1] + 7 * current_time / 100e3
        elif current_time < 200e3:
            bounding_box[:, 0] = bounding_box[:, 0] + 3.5 + 3.5 * (current_time - 100e3) / 100e3
            bounding_box[:, 1] = bounding_box[:, 1] + 7 - 7 * (current_time - 100e3) / 100e3
        elif current_time < 300e3:
            bounding_box[:, 0] = bounding_box[:, 0] + 7 - 7 * (current_time - 200e3) / 100e3
            bounding_box[:, 1] = bounding_box[:, 1]
        else:
            bounding_box[:, 0] = bounding_box[:, 0]
            bounding_box[:, 1] = bounding_box[:, 1]

        bounding_box = np.maximum(bounding_box, np.zeros_like(bounding_box))
        bounding_box[:, 0] = np.minimum(bounding_box[:, 0], np.ones_like(bounding_box[:, 0]) * self.width - 1)
        bounding_box[:, 1] = np.minimum(bounding_box[:, 1], np.ones_like(bounding_box[:, 1]) * self.height - 1)

        return bounding_box


class NCars(NCaltech101):
    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='training',
                 event_representation='histogram', shuffle=True):
        """
        Creates an iterator over the N_Caltech101 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        if mode == 'training':
            mode = 'train'
        elif mode == 'testing':
            mode = 'test'
        if mode == 'validation':
            mode = 'val'
        self.root = os.path.join(root, mode)
        self.object_classes = object_classes
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.event_representation = event_representation

        self.files = listdir(self.root)
        self.nr_samples = len(self.files)

        if shuffle:
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = np.loadtxt(os.path.join(self.root, self.files[idx], 'is_car.txt')).astype(np.int64)
        events = np.loadtxt(os.path.join(self.root, self.files[idx], 'events.txt'), dtype=np.float32)
        events[events[:, -1] == 0, -1] = -1
        nr_events = events.shape[0]

        window_start = 0
        window_end = nr_events
        if self.augmentation:
            events = random_shift_events(events, max_shift=10,resolution=(self.height, self.width))
            events = random_flip_events_along_x(events, resolution=(self.height, self.width))
            window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))

        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end = min(nr_events, window_start + self.nr_events_window)

        events = events[window_start:window_end, :]

        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, label, histogram