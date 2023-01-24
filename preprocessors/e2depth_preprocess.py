import torch
import torch.nn as nn

class EventPreprocessor:
    """
    Utility class to preprocess event tensors.
    Can perform operations such as hot pixel removing, event tensor normalization,
    or flipping the event tensor.
    """

    def __init__(self, options):

        print('== Event preprocessing ==')
        self.no_normalize = options.no_normalize
        if self.no_normalize:
            print('!!Will not normalize event tensors!!')
        else:
            print('Will normalize event tensors.')

        self.hot_pixel_locations = []
        self.hot_pixel_mask = None
        if options.hot_pixels_file:
            try:
                self.hot_pixel_locations = np.loadtxt(options.hot_pixels_file, delimiter=',').astype(np.int)
                print('Will remove {} hot pixels'.format(self.hot_pixel_locations.shape[0]))
            except IOError:
                print('WARNING: could not load hot pixels file: {}'.format(options.hot_pixels_file))

        self.flip = options.flip
        if self.flip:
            print('Will flip event tensors.')

    def __call__(self, events):

        # Initialize hot pixel mask if not already initialized
        if len(self.hot_pixel_locations) > 0 and self.hot_pixel_mask is None:
            self.hot_pixel_mask = torch.ones_like(events, device=events.device)
            for x, y in self.hot_pixel_locations:
                self.hot_pixel_mask[:, :, y, x] = 0

        # Remove (i.e. zero out) the hot pixels
        if self.hot_pixel_mask is not None:
            events = events * self.hot_pixel_mask

        # Flip tensor vertically and horizontally
        if self.flip:
            events = torch.flip(events, dims=[2, 3])

        # Normalize the event tensor (voxel grid) so that
        # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
        if not self.no_normalize:
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                # compute mean and stddev of the **nonzero** elements of the event tensor
                # we do not use PyTorch's default mean() and std() functions since it's faster
                # to compute it by hand than applying those funcs to a masked array

                mean = torch.sum(events, dtype=torch.float32) / num_nonzeros  # force torch.float32 to prevent overflows when using 16-bit precision
                stddev = torch.sqrt(torch.sum(events ** 2, dtype=torch.float32) / num_nonzeros - mean ** 2)
                mask = nonzero_ev.type_as(events)
                events = mask * (events - mean) / stddev

        return events


def get_preprocessor():
    class options:
        hot_pixels_file=None
        flip = False
        no_normalize=True
        no_recurrent=False
        use_gpu=True
        output_folder=None
    event_preprocessor = EventPreprocessor(options)
    pad = nn.ReflectionPad2d((3, 3, 2, 2))
    return lambda data: [pad(event_preprocessor(data)).to('cuda'), None]#, memory_format=memory_format)


preprocess = get_preprocessor()    


