import torch
import cv2
import numpy as np
import matplotlib

def visualize(tensor, apply_colormap=True):
    tensor = tensor.sum(-3) # compress the chanel simension
    tensor = tensor.reshape([tensor.shape[-2], tensor.shape[-1]])
    tensor = tensor * 255
    tensor = tensor.cpu().numpy()
    tensor = tensor.astype('uint8')
    if apply_colormap:
        tensor = cv2.applyColorMap(tensor, cv2.COLORMAP_JET)
    cv2.imshow('image', tensor)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()



def visualize_depth(tensor, apply_colormap=True):
    tensor = tensor.sum(-3) # compress the chanel simension
    tensor = tensor.reshape([tensor.shape[-2], tensor.shape[-1]])
    tensor = tensor * 255
    tensor = tensor.cpu().numpy()
    tensor = tensor.astype('uint8')
    if apply_colormap:
        tensor = cv2.applyColorMap(tensor, cv2.COLORMAP_JET)
    cv2.imshow('image', tensor)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()



#  borrowed code
def events_to_image(inp_events, color_scheme="green_red"):
    """
    Visualize the input events.
    :param inp_events: [batch_size x 2 x H x W] per-pixel and per-polarity event count
    :param color_scheme: green_red/gray
    :return event_image: [H x W x 3] color-coded event image
    """
    pos = inp_events[:, :, 0]
    neg = inp_events[:, :, 1]
    pos_max = np.percentile(pos, 99)
    pos_min = np.percentile(pos, 1)
    neg_max = np.percentile(neg, 99)
    neg_min = np.percentile(neg, 1)
    max = pos_max if pos_max > neg_max else neg_max

    if pos_min != max:
        pos = (pos - pos_min) / (max - pos_min)
    if neg_min != max:
        neg = (neg - neg_min) / (max - neg_min)

    pos = np.clip(pos, 0, 1)
    neg = np.clip(neg, 0, 1)

    event_image = np.ones((inp_events.shape[0], inp_events.shape[1]))
    if color_scheme == "gray":
        event_image *= 0.5
        pos *= 0.5
        neg *= -0.5
        event_image += pos + neg

    elif color_scheme == "green_red":
        event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
        event_image *= 0
        mask_pos = pos > 0
        mask_neg = neg > 0
        mask_not_pos = pos == 0
        mask_not_neg = neg == 0

        event_image[:, :, 0][mask_pos] = 0
        event_image[:, :, 1][mask_pos] = pos[mask_pos]
        event_image[:, :, 2][mask_pos * mask_not_neg] = 0
        event_image[:, :, 2][mask_neg] = neg[mask_neg]
        event_image[:, :, 0][mask_neg] = 0
        event_image[:, :, 1][mask_neg * mask_not_pos] = 0

    return event_image







# floaw to image
def flow_to_image(flow_x, flow_y):
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)
    min_mag = np.min(mag)
    mag_range = np.max(mag) - min_mag

    ang = np.arctan2(flow_y, flow_x) + np.pi
    ang *= 1.0 / np.pi / 2.0

    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3])
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 1.0
    hsv[:, :, 2] = mag - min_mag
    if mag_range != 0.0:
        hsv[:, :, 2] /= mag_range

    flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return (255 * flow_rgb).astype(np.uint8)

def visualize_flow(flow):
    height, width = flow.shape[2], flow.shape[3]
    flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
    flow_npy = flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
    flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Estimated Flow", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated Flow", flow_npy)
    cv2.waitKey(100)


if __name__ == '__main__':
    for i in range(1000):
        img = torch.randn(1, 2, 256, 346)
        visualize_flow(img)







