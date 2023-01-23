import torch
import torch.nn.functional as F
from ._C.c_ops import activation_increment, conv_template
from .allowed_configs import conv2d as conv2d_config


def activation_incr(x, input_incr):
    output_= torch.empty(x.shape, dtype=torch.float, device='cuda')
    activation_increment(x, input_incr, output_)
    return output_



# filter dimensions: [cout, kx, ky, cin]
def functional_conv_module(x_incr, conv_weights, mask=None, stride=(1,1), padding=(1,1)):   
    batch = x_incr.shape[0]
    out_H = int((x_incr.shape[2] + 2*padding[0] - conv_weights.shape[1] ) // stride[0] + 1)
    out_W = int((x_incr.shape[3] + 2*padding[1] - conv_weights.shape[2] ) // stride[1] + 1)
    out_C = conv_weights.shape[3]
    # return [output_, None]
    filter_size = conv_weights.shape[-1]
    conv_config = {
        'batch': batch,
        'k': filter_size,
        'padding': padding[0],
        'stride': stride[0],
    }
    if conv_config in conv2d_config:
        x_padded = F.pad(x_incr, (padding[0], padding[0], padding[1], padding[1]), mode='constant', value=0)
        if mask == None:
            mask = F.max_pool2d(x_padded, kernel_size=filter_size, stride=stride, padding=0, return_indices=False, ceil_mode=False)
        output_= torch.empty((batches, out_C, out_H, out_W), dtype=torch.float, device='cuda', memory_format=torch.channels_last)
        conv_template(x_padded, mask, conv_weights, output_,filter_size)
        return [output_, None]
    else:
        print('fallback')
        output_ = F.conv2d(x_incr[0], conv_weights, bias=None, padding=padding, stride=stride)
        return [output_, None]


