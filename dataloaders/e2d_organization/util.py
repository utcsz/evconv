import numpy as np
from math import fabs

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def robust_min(img, p=5):
    return np.percentile(img.ravel(), p)


def robust_max(img, p=95):
    return np.percentile(img.ravel(), p)


def normalize(img, m=10, M=90):
    return np.clip((img - robust_min(img, m)) / (robust_max(img, M) - robust_min(img, m)), 0.0, 1.0)


def first_element_greater_than(values, req_value):
    """Returns the pair (i, values[i]) such that i is the minimum value that satisfies values[i] >= req_value.
    Returns (-1, None) if there is no such i.
    Note: this function assumes that values is a sorted array!"""
    i = np.searchsorted(values, req_value)
    if i >= len(values) or abs(values[i] - req_value) > 0.01:
        i = i - 1
    val = values[i] if i < len(values) else None
    return (i, val)


def last_element_less_than(values, req_value):
    """Returns the pair (i, values[i]) such that i is the maximum value that satisfies values[i] <= req_value.
    Returns (-1, None) if there is no such i.
    Note: this function assumes that values is a sorted array!"""
    i = np.searchsorted(values, req_value, side='right') - 1
    val = values[i] if i >= 0 else None
    return (i, val)


def closest_element_to(values, req_value):
    """Returns the tuple (i, values[i], diff) such that i is the closest value to req_value,
    and diff = |values(i) - req_value|
    Note: this function assumes that values is a sorted array!"""
    assert(len(values) > 0)

    i = np.searchsorted(values, req_value, side='left')
    if i > 0 and (i == len(values) or fabs(req_value - values[i - 1]) < fabs(req_value - values[i])):
        idx = i - 1
        val = values[i - 1]
    else:
        idx = i
        val = values[i]

    diff = fabs(val - req_value)
    return (idx, val, diff)


def event_histogram(events, W, H):
    t, x, y, p = events.T
    x = x.astype(np.int)
    y = y.astype(np.int)

    img_pos = np.zeros((H * W,), dtype="float32")
    img_neg = np.zeros((H * W,), dtype="float32")

    np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
    np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)

    histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))

    return histogram    


def events_to_voxel_grid_absT(events, num_bins, height, width, deltaT, startT):
    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if events.shape[0] == 0:
        return np.reshape(voxel_grid, (num_bins, height, width))

    events[:, 0] = (num_bins - 1) * (events[:, 0]/1e9 - startT) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    return voxel_grid


def events_to_voxel_grid(events, num_bins, height, width):

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if events.shape[0] == 0:
        return np.reshape(voxel_grid, (num_bins, height, width))
    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def normalize_voxelgrid(event_tensor):
    mask = np.nonzero(event_tensor)
    if mask[0].size > 0:
        mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
        if stddev > 0:
            event_tensor[mask] = (event_tensor[mask] - mean) / stddev
    return event_tensor

