import torch
import torch.nn as nn
import torch.nn.functional as F
from .masked_types import DenseT, Masked
from .incr_modules import IncrementReserve, nnBatchNorm2dIncr, nnConvIncr, nnReluIncr, nnReservedActivation, nnReservedMultiplication, nnSigmoidIncr, nnTanhIncr
from .incr_functional import interpolate_from_module

# does not include a sparse version
class ConvLayerIncr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(ConvLayerIncr, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nnConvIncr(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            op = getattr(torch, activation, 'relu')
            self.activation = nnReservedActivation(op)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nnBatchNorm2dIncr(out_channels)
        elif norm == 'IN':
            raise NotImplementedError
            

    # fully connected implementation
    def forward(self, x_incr: Masked) -> Masked:
        out_incr = self.conv2d(x_incr)
        # out_incr = self.kf(out_incr)
        if self.norm == ['BN', 'IN']:
            out_incr = self.norm_layer(out_incr)

        if self.activation is not None:
            out_incr = self.activation(out_incr)

        return out_incr

    def forward_refresh_reservoirs(self, x: DenseT):
        out = self.conv2d.forward_refresh_reservoirs(x)
        if self.norm in ['BN', 'IN']:
            out = self.norm_layer.forward_refresh_reservoirs(out)

        if self.activation is not None:
            out = self.activation.forward_refresh_reservoirs(out)

        return out


class ConvLSTMIncr(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, prev_c_res: IncrementReserve = None):
        super(ConvLSTMIncr, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        
        self.zero_tensors = {}
        self.zero_tensors_incr = {}

        # convfilter:
        self.Gates = nnConvIncr(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

        self.m1 = nnReservedMultiplication()
        self.m2 = nnReservedMultiplication()
        self.m3 = nnReservedMultiplication()
        self.sigmoid1 = nnSigmoidIncr()
        self.sigmoid2 = nnSigmoidIncr()
        self.sigmoid3 = nnSigmoidIncr()
        self.tanh1 = nnTanhIncr()
        self.tanh2 = nnTanhIncr()


    def forward(self, input_incr: Masked, prev_state_incr: Masked):

        # get batch and spatial sizes
        batch_size = input_incr[0].data.size()[0]
        spatial_size = input_incr[0].data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_incr is None:
            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors_incr:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors_incr[state_size] = (
                    [torch.zeros(state_size, device=input_incr[0].device).to(memory_format=torch.channels_last), None],
                    [torch.zeros(state_size, device=input_incr[0].device).to(memory_format=torch.channels_last), None]
                )

            prev_state_incr = self.zero_tensors_incr[state_size]

        prev_h_incr, prev_c_incr = prev_state_incr

        stacked_inputs_incr = torch.cat((input_incr[0], prev_h_incr[0]), 1), None
        gates_incr = self.Gates(stacked_inputs_incr)

        in_gate, remember_gate, out_gate, cell_gate = gates_incr[0].chunk(4, 1)

        # set of pointwise nonlinear operations: output is guaranteed to be sparse
        in_gate = self.sigmoid1([in_gate, None])
        remember_gate = self.sigmoid2([remember_gate, None])
        out_gate = self.sigmoid3([out_gate, None])

        # apply tanh non linearity
        cell_gate = self.tanh1([cell_gate, None])

        cell = self.m1(in_gate, cell_gate) + self.m2(remember_gate, prev_c_incr)

        # nonlinear operation: incrmenet as well as a 
        cell_tanh = self.tanh2(cell)
        hidden = self.m3(out_gate, cell_tanh)    
#        hidden_incr = out_gate_incr * self.cell_incr_tanh(cell_incr)

        return hidden, cell


    # update all reservoirs in the meanwhile
    # call forward_refresh_reservoirs for each sub-increment module
    # same as before but now it's forward_refresh instead, and update_accumulate instead
    def forward_refresh_reservoirs(self, input_, prev_state):
        # get batch and spatial sizes

        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, device=input_.device).to(memory_format=torch.channels_last),
                    torch.zeros(state_size, device=input_.device).to(memory_format=torch.channels_last)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_h, prev_c = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_h), 1)
        gates = self.Gates.forward_refresh_reservoirs(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = self.sigmoid1.forward_refresh_reservoirs(in_gate)
        remember_gate = self.sigmoid2.forward_refresh_reservoirs(remember_gate)
        out_gate = self.sigmoid3.forward_refresh_reservoirs(out_gate)

        # apply tanh non linearity
        cell_gate = self.tanh1.forward_refresh_reservoirs(cell_gate)

        cell = self.m1.forward_refresh_reservoirs(in_gate, cell_gate) + self.m2.forward_refresh_reservoirs(remember_gate, prev_c)

        # compute current cell and hidden state
        cell_tanh = self.tanh2.forward_refresh_reservoirs(cell)
        hidden = self.m3.forward_refresh_reservoirs(out_gate, cell_tanh)
        return hidden, cell


class RecurrentConvLayerIncr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    recurrent_block_type='convlstm', activation='relu', norm=None):
        super().__init__()
        self.conv = ConvLayerIncr(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        if recurrent_block_type == 'convlstm':
            self.recurrent_block = ConvLSTMIncr(input_size=out_channels, hidden_size=out_channels, kernel_size=3)
        else:
            raise NotImplementedError


    def forward(self, x_incr, prev_state_incr):
        x_incr = self.conv(x_incr)
        state_incr = self.recurrent_block(x_incr, prev_state_incr) 
        x_incr = state_incr[0]
        return x_incr, state_incr

    def forward_refresh_reservoirs(self, x: DenseT, prev_state):
        x = self.conv.forward_refresh_reservoirs(x)
        state = self.recurrent_block.forward_refresh_reservoirs(x, prev_state)
        x = state[0]
        return x, state

# took out the instance norm terms: for default clases don't change init. that's how you preserve load_state_dict
class ResidualBlockIncr(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None):
        super().__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nnConvIncr(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nnBatchNorm2dIncr(out_channels)
            self.bn2 = nnBatchNorm2dIncr(out_channels)
        elif norm == 'IN':
            raise NotImplementedError
        self.relu1 = nnReluIncr()

        self.conv2 = nnConvIncr(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu2 = nnReluIncr()

        self.downsample = downsample

    def forward(self, x_incr: Masked) -> Masked:
        residual_incr = x_incr
        out_incr = self.conv1(x_incr)
        if self.norm in ['BN', 'IN']:
            out_incr = self.bn1(out_incr)
        out_incr = self.relu1(out_incr)
        out_incr = self.conv2(out_incr)
        if self.norm in ['BN', 'IN']:
            out_incr = self.bn2(out_incr)

        if self.downsample:
            residual_incr = self.downsample(x_incr)

        out_incr = (out_incr[0] + residual_incr[0], None) # todo : check accumulation mask of doing this way
        out_incr = self.relu2(out_incr)

        return out_incr

    def forward_refresh_reservoirs(self, x):
        residual = x
        out = self.conv1.forward_refresh_reservoirs(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu1.forward_refresh_reservoirs(out)
        out = self.conv2.forward_refresh_reservoirs(out)        
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample.forward_refresh_reservoirs(x)

        out += residual
        out = self.relu2.forward_refresh_reservoirs(out)
        return out


class TransposedConvLayerIncr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super().__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nnConvTranspose2dIncr(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            op = getattr(torch, activation, 'relu')
            self.activation = nnReservedActivation(op)

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)


    def forward(self, x_incr: Masked) -> Masked:
        out_incr = self.transposed_conv2d(x_incr) 

        if self.norm in ['BN', 'IN']:
            out_incr = self.norm_layer(out_incr)

        if self.activation is not None:
            out_incr = self.activation(out_incr)
        return out_incr

    def forward_refresh_reservoirs(self, x):
        out = self.transposed_conv2d.forward_refresh_reservoirs(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out_act = self.activation.forward_refresh_reservoirs(out)
            out = out_act
            
        return out


class UpsampleConvLayerIncr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super().__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nnConvIncr(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            op = getattr(torch, activation, 'relu')
            self.activation = nnReservedActivation(op)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nnBatchNorm2dIncr(out_channels)
        elif norm == 'IN':
            raise NotImplementedError


    def forward(self, x_incr: Masked) -> Masked:
        x_upsampled_incr = interpolate_from_module(x_incr)
        out_incr = self.conv2d(x_upsampled_incr)

        if self.norm in ['BN', 'IN']:
            out_incr = self.norm_layer(out_incr)

        if self.activation is not None:
            out_incr = self.activation(out_incr)
        return out_incr

    def forward_refresh_reservoirs(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d.forward_refresh_reservoirs(x_upsampled)
        if self.norm in ['BN', 'IN']:
            out = self.norm_layer.forward_refresh_reservoirs(out)

        if self.activation is not None:
            out = self.activation.forward_refresh_reservoirs(out)
        return out


class BaseUNetIncr(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = lambda x1,x2: x1+x2
        self.norm = norm

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayerIncr
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayerIncr

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        op = getattr(torch, activation, 'relu')
        self.activation = nnReservedActivation(op)


    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlockIncr(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayerIncr(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)


class UNetRecurrentIncr(BaseUNetIncr):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super().__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = ConvLayerIncr(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayerIncr(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    recurrent_block_type=recurrent_block_type,
                                                    norm=self.norm))
        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()


    def forward(self, x_incr: Masked, prev_states_incr: Masked):

        # print_sparsity(x_incr[0], "before convhead")
        x_incr = self.head(x_incr)
        head = x_incr
        
        # print_sparsity(x_incr[0], "after convhead")

        if prev_states_incr is None:
            prev_states_incr = [None] * self.num_encoders

        # encoder
        blocks = []
        states_incr = []
        for i, encoder in enumerate(self.encoders):
            x_incr, state_incr = encoder(x_incr, prev_states_incr[i])
            blocks.append(x_incr)
            states_incr.append(state_incr)
            # print_sparsity(x_incr[0], "after encoder{}".format(i) )


        # residual blocks
        for i,resblock in enumerate(self.resblocks):
            x_incr = resblock(x_incr)
            # print_sparsity(x_incr[0], "after resblock{}".format(i) )

        # decoder
        for i, decoder in enumerate(self.decoders):
            x_incr = decoder(self.apply_skip_connection(x_incr, blocks[self.num_encoders - i - 1]))
            # print_sparsity(x_incr[0], "after decoder{}".format(i) )

        # tail
        pred_incr = self.pred(self.apply_skip_connection(x_incr, head))
        pred_incr = self.activation(pred_incr)

        # print_sparsity(pred_incr[0], "final")

        return pred_incr, states_incr


    def forward_refresh_reservoirs(self, x: DenseT, prev_states):
        x = self.head.forward_refresh_reservoirs(x)
        head = x
        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder.forward_refresh_reservoirs(x, prev_states[i])
            blocks.append(x)
            states.append(state)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock.forward_refresh_reservoirs(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder.forward_refresh_reservoirs(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        pred = self.pred.forward_refresh_reservoirs(self.apply_skip_connection(x, head))
        pred = self.activation.forward_refresh_reservoirs(pred)

        return pred, states


