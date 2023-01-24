from typing import OrderedDict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
# from incr_modules.utils import count_ops, print_sparsity

from .masked_types import Masked, DenseT
from .incr_functional import IncrPointwiseMultiply, IncrementReserve, bn2d_from_module, conv2d_from_module


# input output increments: interface for incremental modules! 
class IncrementMaskModule(nn.Module):

    # called at the end of non-increment forward pass, if needed.
    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT: 
        return x


# filter: only allow significant elements to pass through
class KFencedMaskModule(IncrementMaskModule):
    def __init__(self, tp: float=.1):
        super().__init__()
        # internally defined reserves
        # self.in_reserve = IncrementReserve()
        self.delta = IncrementReserve() # input tensor dimensions: dense tensor
        self.tp = tp
        self.k = tp

    def floor_by_k(self, T: Masked) -> Masked:  # TODO: implement for sparse inputs.
        return (self.k*torch.floor(0.5 + T/self.k))  

    # accumulate operations: sparsed
    def forward(self, incr: Masked) -> Masked:
        # return incr
        T1 = self.delta.reservoir + incr[0]                 # critical path; could be sparse
        f_delta = self.floor_by_k(T1)                   # critical path; could be sparse
        self.delta.update_reservoir(T1 - f_delta)                       # out of order; sparse
        # self.in_reserve.accumulate(f_delta)         # out of order; sparse: conditional: doesn't need to be done
        return f_delta, None

    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT:
        self.k = self.tp*torch.norm(x)
        # self.in_reserve.update_reservoir(x)
        self.delta.update_reservoir(torch.zeros_like(x))
        return x



class PointwiseMultiplyIncr(IncrementMaskModule):
    def __init__(self, x1res_module: IncrementReserve, x2res_module: IncrementReserve):
        super().__init__()
        # define reserves:
        self.x1_res: IncrementReserve = x1res_module
        self.x2_res: IncrementReserve = x2res_module


    def forward(self, x1_incr: Masked, x2_incr: Masked) -> Masked:
        output_incr = IncrPointwiseMultiply(x1_incr, self.x1_res, x2_incr, self.x2_res)
        return output_incr

    def forward_refresh_reservoirs(self, x1: DenseT, x2: DenseT) -> DenseT:
        return x1*x2


class nnReservedMultiplication(PointwiseMultiplyIncr):
    def __init__(self):
        super().__init__(IncrementReserve(), IncrementReserve())

    def forward(self, x1_incr: Masked, x2_incr: Masked) -> Masked:
        out_incr = PointwiseMultiplyIncr.forward(self, x1_incr, x2_incr)
        self.x1_res.accumulate(x1_incr)
        self.x2_res.accumulate(x2_incr)
        return out_incr

    def forward_refresh_reservoirs(self, x1: DenseT, x2: DenseT) -> DenseT:
        out = PointwiseMultiplyIncr.forward_refresh_reservoirs(self, x1, x2)
        self.x1_res.update_reservoir(x1)
        self.x2_res.update_reservoir(x2)        
        return out



# nonlinear operations which are point ops; dense modules only
class NonlinearPointOpIncr(IncrementMaskModule):
    def __init__(self, res_in: IncrementReserve, op=torch.tanh):
        super().__init__()
        self.reservoir_in = res_in
        self.op = op

    def forward(self, x_incr: Masked) -> Masked:
        # compute only for these inputs.
        return self._nonlin(x_incr)

    # x is like an input: don't update external reservoirs
    def forward_refresh_reservoirs(self, x: DenseT):
        return self.op(x)

    # need to be replaced with c++ binding
    def _nonlin(self, x_incr: Masked):
        output_incr = self.op(self.reservoir_in.reservoir + x_incr[0]) - self.op(self.reservoir_in.reservoir)
        return [output_incr, x_incr[1]]


class nnLinearIncr(nn.Linear):
    
    # fully connected implementation
    def forward(self, x_incr: Masked) -> Masked:
        # print("tot, nz = ", x_incr[0].numel(), (x_incr[0]>=0.001).count_nonzero())
        # return self.weight*x_incr[0], None
        out = F.linear(x_incr[0], self.weight, bias=None)
        return out, None

    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT:
        return super().forward(x)

        # return F.linear(x, self.weight, self.bias)


#linear module
class nnConvIncrBase(nn.Conv2d):

    # def __init__(self,
    #              in_channels: int,
    #              out_channels: int,
    #              kernel_size: Tuple[int, ...],
    #              stride: Tuple[int, ...]=1,
    #              padding: Tuple[int, ...]=0,
    #              bias=None):
    #     nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        # self.conv2d_weights = pf.convert_filter_out_channels_last(self.conv2d.weight).cuda()


    def forward(self, x_incr):
        # print('x_incr: ', x_incr.shape)
        # out = F.conv2d(x_incr[0], self.weight, bias=None, padding=self.padding, stride=self.stride)
        # return out, torch.ones_like(out, dtype=bool)
        return conv2d_from_module(x_incr, self.conv2d_weights)

    def forward_refresh_reservoirs(self, x):
        return super().forward(x)


#linear module
class nnConvIncr(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...]=1,
                 padding: Tuple[int, ...]=0,
                 bias=True):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.kf = KFencedMaskModule(0.01)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]',
                        strict: bool = True):
        nn.Conv2d.load_state_dict(self, state_dict, strict=strict)

    def forward(self, x_incr):
        x_incr = self.kf(x_incr)
        return conv2d_from_module(x_incr, self.weight, stride=self.stride, padding=self.padding)


    def forward_refresh_reservoirs(self, x):
        x = self.kf.forward_refresh_reservoirs(x)
        return super().forward(x)


#linear module
class nnBatchNorm2dIncr(nn.BatchNorm2d):

    def forward(self, x_incr):
        return bn2d_from_module(x_incr, self)


    def forward_refresh_reservoirs(self, x):
        return super().forward(x)


class nnMaxPool2dIncr(nn.MaxPool2d):

    def __init__(self, kernel_size, stride=None, padding=0):
        nn.MaxPool2d.__init__(self, kernel_size=kernel_size, stride=stride, padding=padding)
        self.functional = lambda x: F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.mp_res = IncrementReserve()

    def forward(self, x_incr: Masked) -> Masked:
        out = self.functional(self.mp_res.reservoir+x_incr[0]) - self.functional(self.mp_res.reservoir)
        self.mp_res.accumulate(x_incr)
        return out, None

    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT:
        self.mp_res.update_reservoir(x)
        return super().forward(x)


#linear module
class nnAdaptiveAvgPool2dIncr(nn.AdaptiveAvgPool2d):
    def forward(self, x_incr):
        return nn.AdaptiveAvgPool2d.forward(self, x_incr[0]), None

    def forward_refresh_reservoirs(self, x):
        return super().forward(x)


class nnSequentialIncr(nn.Sequential):
    def forward_refresh_reservoirs(self, x):
        for module in self:
            x = module.forward_refresh_reservoirs(x)
        return x


class nnReservedActivation(NonlinearPointOpIncr):
    def __init__(self, op=torch.relu):
        self.in_res = IncrementReserve() 
        NonlinearPointOpIncr.__init__(self, self.in_res, op)

    def forward(self, x_incr: Masked):
        out = NonlinearPointOpIncr.forward(self, x_incr)
        self.in_res.accumulate(x_incr)
        return out

    def forward_refresh_reservoirs(self, x: DenseT):
        out = NonlinearPointOpIncr.forward_refresh_reservoirs(self, x)
        self.in_res.update_reservoir(x)
        return out


class nnReluIncr(nnReservedActivation):
    def __init__(self):
        nnReservedActivation.__init__(self, torch.relu)

class nnSigmoidIncr(nnReservedActivation):
    def __init__(self):
        nnReservedActivation.__init__(self, torch.sigmoid)

class nnTanhIncr(nnReservedActivation):
    def __init__(self):
        nnReservedActivation.__init__(self, torch.tanh)


