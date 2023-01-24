import torch
import torch.nn as nn
import torch.nn.functional as F
from .maskincr_ops.functional import functional_conv_module
from .masked_types import DenseT, Masked


# singleton class; not the best way but whatev
class AccumStreamManager:
    instance = None
    @classmethod
    def createAccumStream(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def __init__(self):
        self.s = torch.cuda.Stream()
    
    def get_stream(self):
        return self.s


# accumulates inputs: have to make this conditional
class IncrementReserve:
    def __init__(self, x_init = None):
        self.accum_stream = AccumStreamManager.createAccumStream() 
        if x_init == None:
            self.reservoir = None
        else:
            self.reservoir = x_init.clone().detach()
            self._adjust_mem_format()

    def _adjust_mem_format(self):
        # TODO: confirm this: all 4-shape tensors in considered networks are NCHW type:
        if len(self.reservoir.shape) == 4:
            self.reservoir = self.reservoir.to(memory_format=torch.channels_last)

    # dense/sparse accumulate accumulate
    def accumulate(self, incr: Masked):
        self.reservoir.add_(incr[0])
        # with torch.cuda.stream(self.accum_stream.get_stream()):
        #     self.reservoir.add_(incr[0])

    def update_reservoir(self, x: DenseT):
        self.reservoir = x.clone().detach() # not in place right now :(
        self._adjust_mem_format()


def IncrPointwiseMultiply(x1_incr: Masked, x1: IncrementReserve, x2_incr: Masked, x2: IncrementReserve) -> Masked:
    return [(x2_incr[0] + x2.reservoir)*x1_incr[0] + x1.reservoir*x2_incr[0], None]
    # return [x1_incr[0]*x2_incr[0] + x2.reservoir*x1_incr[0] + x1.reservoir*x2_incr[0], x1_incr[1]|x2_incr[1]]


def conv2d_from_module(x: Masked, conv_weights, stride=(1,1), padding=(1, 1)) -> Masked:
    return functional_conv_module(x[0], conv_weights, mask_load=None, mask_compute=x[1], stride=stride, padding=padding)


def transposed_conv2d_from_module(x: Masked, gates: nn.ConvTranspose2d, bias=True) -> Masked:
    return F.conv_transpose2d(x[0], gates.weight, bias=None, stride=gates.stride, \
            padding=gates.padding, output_padding=gates.output_padding, dilation=gates.dilation, groups=gates.groups)


def bn2d_from_module(x: Masked, bnm: nn.BatchNorm2d) -> Masked:
    out1 = F.batch_norm(x[0], running_mean=torch.zeros_like(bnm.running_mean), \
        running_var=bnm.running_var, weight=bnm.weight, training=False, momentum=bnm.momentum, eps=bnm.eps)
    return out1, None


def interpolate_from_module(x: Masked) -> Masked:
    out1 = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=False)
    return out1, None



