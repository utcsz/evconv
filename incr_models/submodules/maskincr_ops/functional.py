import torch
import torch.nn.functional as F
from ._C.c_ops import activation_increment, conv_template
from .allowed_configs import conv2d as conv2d_config


def activation_incr(x, input_incr):
    output_= torch.empty(x.shape, dtype=torch.float, device='cuda')
    activation_increment(x, input_incr, output_)
    return output_


# filter dimensions: [cout, kx, ky, cin]
def functional_conv_module(x_incr, conv_weights, mask_load=None, mask_compute=None, stride=(1,1), padding=(1,1), fallback=False):   
    batch = x_incr.shape[0]
    out_C = conv_weights.shape[0]
    out_H = int((x_incr.shape[2] + 2*padding[0] - conv_weights.shape[2] ) // stride[0] + 1)
    out_W = int((x_incr.shape[3] + 2*padding[1] - conv_weights.shape[3] ) // stride[1] + 1)

    filter_size = conv_weights.shape[-1]
    conv_config = {
        'batch': batch,
        'k': filter_size,
        # 'padding': padding[0],
        'stride': stride[0],
    }
    if fallback or conv_config in conv2d_config:
        x_padded = F.pad(x_incr, padding, mode='constant', value=0).to(memory_format=torch.channels_last)
        if mask_compute == None:
            mask_compute = F.max_pool2d(x_padded, kernel_size=filter_size, stride=stride, padding=0, return_indices=False, ceil_mode=False).to(memory_format=torch.channels_last).to(dtype=torch.int)
        if mask_load == None:
            mask_load = torch.ones_like(x_padded).to(dtype=torch.int, memory_format=torch.channels_last) #F.max_pool2d(x_padded, kernel_size=filter_size, stride=stride, padding=0, return_indices=False, ceil_mode=False)
            # mask_load = torch.max_pool3d(torch.abs(x_padded), (32, 1, 1), ceil_mode=True).to(dtype=torch.int, memory_format=torch.channels_last) #F.max_pool2d(x_padded, kernel_size=filter_size, stride=stride, padding=0, return_indices=False, ceil_mode=False)
        
        output_= torch.empty((batch, out_C, out_H, out_W), dtype=torch.float, device='cuda', memory_format=torch.channels_last)
        conv_template(x_padded, mask_load, mask_compute, conv_weights, output_,filter_size)
        return [output_, None]
    else:
        print('fallback')
        print(conv_config)
        output_ = F.conv2d(x_incr[0], conv_weights, bias=None, padding=padding, stride=stride)
        return [output_, None]



if __name__ == '__main__':
    x_incr=torch.ones(1, 39, 224, 224).cuda()
    conv_weights=torch.randn(32, 39, 3, 3).cuda()
    o1, _ = functional_conv_module(
        x_incr, 
        conv_weights,
        padding=(2,2)
    )

    o2,_ = functional_conv_module(
        x_incr, 
        conv_weights,
        padding=(2,2),
        fallback=True
    )

    print(torch.sum(torch.abs(o1 - o2)))




