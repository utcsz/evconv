from .dataset import NCaltech101
from .dataset import NCars        if object_classes == 'all':
            self.object_classes = listdir(root)
        else:
            self.object_classes = object_classes

        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.event_representation = event_representation

