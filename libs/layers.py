import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.fft as fft
import math
import copy
from functools import partial


def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value


class Identity(nn.Module):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
    not used anymore as
    https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)
        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        return self.id(x)


class Shortcut2d(nn.Module):
    '''
    (-1, in, S, S) -> (-1, out, S, S)
    Used in SimpleResBlock
    '''

    def __init__(self, in_features=None,
                 out_features=None,):
        super(Shortcut2d, self).__init__()
        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x, edge=None, grid=None):
        x = x.permute(0, 2, 3, 1)
        x = self.shortcut(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PositionalEncoding(nn.Module):
    '''
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    This is not necessary if spacial coords are given
    input is (batch, seq_len, d_model)
    '''

    def __init__(self, d_model, 
                       dropout=0.1, 
                       max_len=2**13):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(2**13) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Conv2dResBlock(nn.Module):
    '''
    Conv2d + a residual block
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Modified from ResNet's basic block, one conv less, no batchnorm
    No batchnorm
    '''

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 basic_block=False,
                 activation_type='silu'):
        super(Conv2dResBlock, self).__init__()

        activation_type = default(activation_type, 'silu')
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv2d(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Shortcut2d(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class GraphConvolution(nn.Module):
    """
    A modified implementation from 
    https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    to incorporate batch size, and multiple edge

    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, debug=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.debug = debug
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge):
        if x.size(-1) != self.in_features:
            x = x.transpose(-2, -1).contiguous()
        assert x.size(1) == edge.size(-1)
        support = torch.matmul(x, self.weight)

        support = support.transpose(-2, -1).contiguous()
        output = torch.matmul(edge, support.unsqueeze(-1))

        output = output.squeeze()
        if self.bias is not None:
            return output + self.bias.unsqueeze(-1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphAttention(nn.Module):
    """
    Simple GAT layer, modified from https://github.com/Diego999/pyGAT/blob/master/layers.py
    to incorporate batch size similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features,
                 out_features,
                 alpha=1e-2,
                 concat=True,
                 graph_lap=True,  # graph laplacian may have negative entries
                 interaction_thresh=1e-6,
                 dropout=0.1):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.graph_lap = graph_lap
        self.thresh = interaction_thresh

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        xavier_normal_(self.W, gain=np.sqrt(2.0))

        self.a = nn.Parameter(torch.FloatTensor(2*out_features, 1))
        xavier_normal_(self.a, gain=np.sqrt(2.0))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, node, adj):
        h = torch.matmul(node, self.W)
        bsz, seq_len = h.size(0), h.size(1)

        a_input = torch.cat([h.repeat(1, 1, seq_len).view(bsz, seq_len * seq_len, -1),
                             h.repeat(1, seq_len, 1)], dim=2)
        a_input = a_input.view(bsz, seq_len, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15*torch.ones_like(e)
        if self.graph_lap:
            attention = torch.where(adj.abs() > self.thresh, e, zero_vec)
        else:
            attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' ('\
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class EdgeEncoder(nn.Module):
    def __init__(self, out_dim: int,
                 edge_feats: int,
                 raw_laplacian=None):
        super(EdgeEncoder, self).__init__()
        assert out_dim > edge_feats
        self.return_lap = raw_laplacian
        if self.return_lap:
            out_dim = out_dim - edge_feats

        conv_dim0 = int(out_dim/3*2)
        conv_dim1 = int(out_dim - conv_dim0)
        self.lap_conv1 = Conv2dResBlock(edge_feats, conv_dim0)
        self.lap_conv2 = Conv2dResBlock(conv_dim0, conv_dim1)

    def forward(self, lap):
        edge1 = self.lap_conv1(lap)
        edge2 = self.lap_conv2(edge1)
        if self.return_lap:
            return torch.cat([lap, edge1, edge2], dim=1)
        else:
            return torch.cat([edge1, edge2], dim=1)


class Conv2dEncoder(nn.Module):
    r'''
    old code: first conv then pool
    Similar to a LeNet block
    \approx 1/4 subsampling
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 scaling_factor: int = 2,
                 residual=False,
                 activation_type='silu',
                 debug=False):
        super(Conv2dEncoder, self).__init__()

        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding//2 if padding//2 >= 1 else 1
        padding2 = padding//4 if padding//4 >= 1 else 1
        activation_type = default(activation_type, 'silu')
        self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
                                    padding=padding,
                                    residual=residual)
        self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
                                    padding=padding1,
                                    stride=stride, residual=residual)
        self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding2, residual=residual)
        self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
                                    kernel_size=kernel_size,
                                    residual=residual)
        self.pool0 = nn.AvgPool2d(kernel_size=scaling_factor,
                                  stride=scaling_factor)
        self.pool1 = nn.AvgPool2d(kernel_size=scaling_factor,
                                  stride=scaling_factor)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        # self.activation = nn.LeakyReLU() # leakyrelu decreased performance 10 times?
        self.debug = debug

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.activation(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.pool1(out)
        out = self.activation(out)
        return out


# class Interp2dEncoder(nn.Module):
#     r'''
#     Using Interpolate instead of avg pool
#     interp dim hard coded or using a factor
#     '''

#     def __init__(self, in_dim: int,
#                  out_dim: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  padding: int = 1,
#                  dilation: int = 1,
#                  interp_size=None,
#                  residual=False,
#                  activation_type='silu',
#                  dropout=0.1,
#                  debug=False):
#         super(Interp2dEncoder, self).__init__()

#         conv_dim0 = out_dim // 3
#         conv_dim1 = out_dim // 3
#         conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
#         padding1 = padding//2 if padding//2 >= 1 else 1
#         padding2 = padding//4 if padding//4 >= 1 else 1
#         activation_type = default(activation_type, 'silu')
#         self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
#                                     padding=padding, activation_type=activation_type,
#                                     dropout=dropout,
#                                     residual=residual)
#         self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
#                                     padding=padding1,
#                                     stride=stride, residual=residual,
#                                     dropout=dropout,
#                                     activation_type=activation_type,)
#         self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
#                                     dilation=dilation,
#                                     padding=padding2, residual=residual,
#                                     dropout=dropout,
#                                     activation_type=activation_type,)
#         self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
#                                     kernel_size=kernel_size,
#                                     residual=residual,
#                                     dropout=dropout,
#                                     activation_type=activation_type,)
#         if isinstance(interp_size[0], float) and isinstance(interp_size[1], float):
#             self.interp0 = lambda x: F.interpolate(x, scale_factor=interp_size[0],
#                                                    mode='bilinear',
#                                                    recompute_scale_factor=True,
#                                                    align_corners=True)
#             self.interp1 = lambda x: F.interpolate(x, scale_factor=interp_size[1],
#                                                    mode='bilinear',
#                                                    recompute_scale_factor=True,
#                                                    align_corners=True,)
#         elif isinstance(interp_size[0], tuple) and isinstance(interp_size[1], tuple):
#             self.interp0 = lambda x: F.interpolate(x, size=interp_size[0],
#                                                    mode='bilinear',
#                                                    align_corners=True)
#             self.interp1 = lambda x: F.interpolate(x, size=interp_size[1],
#                                                    mode='bilinear',
#                                                    align_corners=True,)
#         elif interp_size is None:
#             self.interp0 = Identity()
#             self.interp1 = Identity()
#         else:
#             raise NotImplementedError("interpolation size not implemented.")
#         self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
#         # self.activation = nn.LeakyReLU() # leakyrelu decreased performance 10 times?
#         self.add_res = residual
#         self.debug = debug

#     def forward(self, x):

#         x = self.conv0(x)
#         x = self.interp0(x)
#         x = self.activation(x)

#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         out = torch.cat([x1, x2, x3], dim=1)

#         if self.add_res:
#             out += x
#         out = self.interp1(out)
#         out = self.activation(out)
#         return out

class Interp2dEncoder(nn.Module):
    r'''
    Using Interpolate instead of avg pool
    interp dim hard coded or using a factor
    old code uses lambda and cannot be pickled
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 interp_size=None,
                 residual=False,
                 activation_type='silu',
                 dropout=0.1,
                 debug=False):
        super(Interp2dEncoder, self).__init__()

        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding//2 if padding//2 >= 1 else 1
        padding2 = padding//4 if padding//4 >= 1 else 1
        activation_type = default(activation_type, 'silu')
        self.interp_size = interp_size
        self.is_scale_factor = isinstance(
            interp_size[0], float) and isinstance(interp_size[1], float)
        self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
                                    padding=padding, activation_type=activation_type,
                                    dropout=dropout,
                                    residual=residual)
        self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
                                    padding=padding1,
                                    stride=stride, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding2, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
                                    kernel_size=kernel_size,
                                    residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.debug = debug

    def forward(self, x):
        x = self.conv0(x)
        if self.is_scale_factor:
            x = F.interpolate(x, scale_factor=self.interp_size[0],
                              mode='bilinear',
                              recompute_scale_factor=True,
                              align_corners=True)
        else:
            x = F.interpolate(x, size=self.interp_size[0],
                              mode='bilinear',
                              align_corners=True)
        x = self.activation(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        if self.add_res:
            out += x

        if self.is_scale_factor:
            out = F.interpolate(out, scale_factor=self.interp_size[1],
                                mode='bilinear',
                                recompute_scale_factor=True,
                                align_corners=True,)
        else:
            out = F.interpolate(out, size=self.interp_size[1],
                              mode='bilinear',
                              align_corners=True)
        out = self.activation(out)
        return out


class DeConv2dBlock(nn.Module):
    '''
    Similar to a LeNet block
    4x upsampling, dimension hard-coded
    '''

    def __init__(self, in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 stride: int = 2,
                 kernel_size: int = 3,
                 padding: int = 2,
                 output_padding: int = 1,
                 dropout=0.1,
                 activation_type='silu',
                 debug=False):
        super(DeConv2dBlock, self).__init__()
        # assert stride*2 == scaling_factor
        padding1 = padding//2 if padding//2 >= 1 else 1

        self.deconv0 = nn.ConvTranspose2d(in_channels=in_dim,
                                          out_channels=hidden_dim,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          output_padding=output_padding,
                                          padding=padding)
        self.deconv1 = nn.ConvTranspose2d(in_channels=hidden_dim,
                                          out_channels=out_dim,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          output_padding=output_padding,
                                          padding=padding1,  # hard code bad, 1: for 85x85 grid, 2 for 43x43 grid
                                          )
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, x):
        x = self.deconv0(x)
        x = self.dropout(x)

        x = self.activation(x)
        x = self.deconv1(x)
        x = self.activation(x)
        return x


# class Interp2dUpsample(nn.Module):
#     '''
#     interp->conv2d->interp
#     or
#     identity
#     '''

#     def __init__(self, in_dim: int,
#                  out_dim: int,
#                  kernel_size: int = 3,
#                  padding: int = 1,
#                  residual=False,
#                  conv_block=True,
#                  interp_mode='bilinear',
#                  interp_size=None,
#                  activation_type='silu',
#                  dropout=0.1,
#                  debug=False):
#         super(Interp2dUpsample, self).__init__()
#         activation_type = default(activation_type, 'silu')
#         self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         if conv_block:
#             self.conv = nn.Sequential(Conv2dResBlock(
#                 in_dim, out_dim,
#                 kernel_size=kernel_size,
#                 padding=padding,
#                 residual=residual,
#                 dropout=dropout,
#                 activation_type=activation_type),
#                 self.dropout,
#                 self.activation)
#         self.conv_block = conv_block
#         if isinstance(interp_size[0], float) and isinstance(interp_size[1], float):
#             self.interp0 = lambda x: F.interpolate(x, scale_factor=interp_size[0],
#                                                    mode=interp_mode,
#                                                    recompute_scale_factor=True,
#                                                    align_corners=True)
#             self.interp1 = lambda x: F.interpolate(x, scale_factor=interp_size[1],
#                                                    mode=interp_mode,
#                                                    recompute_scale_factor=True,
#                                                    align_corners=True)
#         elif isinstance(interp_size[0], tuple) and isinstance(interp_size[1], tuple):
#             self.interp0 = lambda x: F.interpolate(x, size=interp_size[0],
#                                                    mode=interp_mode,
#                                                    align_corners=True)
#             self.interp1 = lambda x: F.interpolate(x, size=interp_size[1],
#                                                    mode=interp_mode,
#                                                    align_corners=True)
#         elif interp_size is None:
#             self.interp0 = Identity()
#             self.interp1 = Identity()

#         self.debug = debug

#     def forward(self, x):
#         x = self.interp0(x)
#         if self.conv_block:
#             x = self.conv(x)
#         x = self.interp1(x)
#         return x

class Interp2dUpsample(nn.Module):
    '''
    interpolate then Conv2dResBlock
    old code uses lambda and cannot be pickled
    temp hard-coded dimensions
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual=False,
                 conv_block=True,
                 interp_mode='bilinear',
                 interp_size=None,
                 activation_type='silu',
                 dropout=0.1,
                 debug=False):
        super(Interp2dUpsample, self).__init__()
        activation_type = default(activation_type, 'silu')
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(Conv2dResBlock(
                in_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                residual=residual,
                dropout=dropout,
                activation_type=activation_type),
                self.dropout,
                self.activation)
        self.conv_block = conv_block
        self.interp_size = interp_size
        self.interp_mode = interp_mode
        self.debug = debug

    def forward(self, x):
        x = F.interpolate(x, size=self.interp_size[0],
                          mode=self.interp_mode,
                          align_corners=True)
        if self.conv_block:
            x = self.conv(x)
        x = F.interpolate(x, size=self.interp_size[1],
                          mode=self.interp_mode,
                          align_corners=True)
        return x

def attention(query, key, value,
              mask=None, dropout=None, weight=None,
              attention_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''

    d_k = query.size(-1)

    if attention_type == 'cosine':
        p_attn = F.cosine_similarity(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
        seq_len = scores.size(-1)

        if attention_type == 'softmax':
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim=-1)
        elif attention_type in ['fourier', 'integral', 'local']:
            if mask is not None:
                scores = scores.masked_fill(mask == 0, 0)
            p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(p_attn, value)

    return out, p_attn


def linear_attention(query, key, value,
                     mask=None, dropout=None,
                     attention_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.size(-2)
    if attention_type in ['linear', 'global']:
        query = query.softmax(dim=-1)
        key = key.softmax(dim=-2)
    scores = torch.matmul(key.transpose(-2, -1), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(query, p_attn)
    return out, p_attn

def causal_linear_attn(query, key, value, kv_mask = None, dropout = None, eps = 1e-7):
    '''
    Modified from https://github.com/lucidrains/linear-attention-transformer
    '''
    bsz, n_head, seq_len, d_k, dtype = *query.shape, query.dtype

    key /= seq_len

    if kv_mask is not None:
        mask = kv_mask[:, None, :, None]
        key = key.masked_fill_(~mask, 0.)
        value = value.masked_fill_(~mask, 0.)
        del mask
    
    b_q, b_k, b_v = [x.reshape(bsz, n_head, -1, 1, d_k) for x in (query, key, value)]

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim = -2).type(dtype)

    p_attn = torch.einsum('bhund,bhune->bhude', b_k, b_v)
    p_attn = p_attn.cumsum(dim = -3).type(dtype)
    if dropout is not None:
        p_attn = F.dropout(p_attn)

    D_inv = 1. / torch.einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    attn = torch.einsum('bhund,bhude,bhun->bhune', b_q, p_attn, D_inv)
    return attn.reshape(*query.shape), p_attn

class SimpleAttention(nn.Module):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types: 
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = torch.einsum('bhnd,bhne->bhde', k, v)
    attn = torch.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm=False,
                 norm_type='layer',
                 eps=1e-5,
                 debug=False):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.add_norm = norm
        self.norm_type = norm_type
        if norm:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head*pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.size(0)
        if weight is not None:
            query, key = weight*query, weight*key

        query, key, value = \
            [layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
             for layer, x in zip(self.linears, (query, key, value))]

        if self.add_norm:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                value = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                query = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), value.transpose(-2, -1)

        if pos is not None and self.pos_dim > 0:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.n_head, 1, 1])
            query, key, value = [torch.cat([pos, x], dim=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        elif self.attention_type == 'causal':
            assert mask is not None
            x, self.attn_weight = causal_linear_attn(query, key, value,
                                                   mask=mask,
                                                   dropout=self.dropout)
        else:
            x, self.attn_weight = attention(query, key, value,
                                            mask=mask,
                                            attention_type=self.attention_type,
                                            dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
            (self.d_k + self.pos_dim)
        att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight

    def _reset_parameters(self):
        for param in self.linears.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                if self.diagonal_weight > 0.0:
                    param.data += self.diagonal_weight * \
                        torch.diag(torch.ones(
                            param.size(-1), dtype=torch.float))
                if self.symmetric_init:
                    param.data += param.data.T
                    # param.data /= 2.0
            else:
                constant_(param, 0)

    def _get_norm(self, eps):
        if self.attention_type in ['linear', 'galerkin', 'global']:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_V = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
        else:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_Q = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_Q = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])


class FeedForward(nn.Module):
    def __init__(self, in_dim=256,
                 dim_feedforward: int = 1024,
                 out_dim=None,
                 batch_norm=False,
                 activation='relu',
                 dropout=0.1):
        super(FeedForward, self).__init__()
        out_dim = default(out_dim, in_dim)
        n_hidden = dim_feedforward
        self.lr1 = nn.Linear(in_dim, n_hidden)

        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(n_hidden)
        self.lr2 = nn.Linear(n_hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.lr1(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = x.permute((0, 2, 1))
            x = self.bn(x)
            x = x.permute((0, 2, 1))
        x = self.lr2(x)
        return x


class BulkRegressor(nn.Module):
    '''
    Bulk regressor:

    Args:
        - in_dim: seq_len
        - n_feats: pointwise hidden features
        - n_targets: number of overall bulk targets
        - pred_len: number of output sequence length
            in each sequence in each feature dimension (for eig prob this=1)

    Input:
        (-1, seq_len, n_features)
    Output:
        (-1, pred_len, n_target)
    '''

    def __init__(self, in_dim,  # seq_len
                 n_feats,  # number of hidden features
                 n_targets,  # number of frequency target
                 pred_len,
                 n_hidden=None,
                 sort_output=False,
                 dropout=0.1):
        super(BulkRegressor, self).__init__()
        n_hidden = default(n_hidden, pred_len * 4)
        self.linear = nn.Linear(n_feats, n_targets)
        freq_out = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            nn.LeakyReLU(),  # frequency can be localized
            nn.Linear(n_hidden, pred_len),
        )
        self.regressor = nn.ModuleList(
            [copy.deepcopy(freq_out) for _ in range(n_targets)])
        self.dropout = nn.Dropout(dropout)
        self.sort_output = sort_output

    def forward(self, x):
        x = self.linear(x)
        x = x.transpose(-2, -1).contiguous()
        out = []
        for i, layer in enumerate(self.regressor):
            out.append(layer(x[:, i, :]))  # i-th target predict
        x = torch.stack(out, dim=-1)
        x = self.dropout(x)
        if self.sort_output:
            x, _ = torch.sort(x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 n_grid=None,
                 dropout=0.1,
                 return_freq=False,
                 activation='silu',
                 debug=False):
        super(SpectralConv1d, self).__init__()

        '''
        Modified Zongyi Li's Spectral1dConv code
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
        '''

        self.linear = nn.Linear(in_dim, out_dim)  # for residual
        self.modes = modes
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.n_grid = n_grid  # just for debugging
        self.fourier_weight = Parameter(
            torch.FloatTensor(in_dim, out_dim, modes, 2))
        xavier_normal_(self.fourier_weight, gain=1/(in_dim*out_dim))
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_1d(a, b):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        op = partial(torch.einsum, "bix,iox->box")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        '''
        Input: (-1, n_grid, in_features)
        Output: (-1, n_grid, out_features)
        '''
        seq_len = x.size(1)
        res = self.linear(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x_ft = fft.rfft(x, n=seq_len, norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = self.complex_matmul_1d(
            x_ft[:, :, :self.modes], self.fourier_weight)

        pad_size = seq_len//2 + 1 - self.modes
        out_ft = F.pad(out_ft, (0, 0, 0, pad_size), "constant", 0)

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft(out_ft, n=seq_len, norm="ortho")

        x = x.permute(0, 2, 1)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 n_grid=None,
                 dropout=0.1,
                 norm='ortho',
                 activation='silu',
                 return_freq=False,  # whether to return the frequency target
                 debug=False):
        super(SpectralConv2d, self).__init__()

        '''
        Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
        using only real weights
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)  # for residual
        self.modes = modes
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.n_grid = n_grid  # just for debugging
        self.fourier_weight = nn.ParameterList([Parameter(
            torch.FloatTensor(in_dim, out_dim,
                                                modes, modes, 2)) for _ in range(2)])
        for param in self.fourier_weight:
            xavier_normal_(param, gain=1/(in_dim*out_dim)
                           * np.sqrt(in_dim+out_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = norm
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        '''
        Input: (-1, n_grid**2, in_features) or (-1, n_grid, n_grid, in_features)
        Output: (-1, n_grid**2, out_features) or (-1, n_grid, n_grid, out_features)
        '''
        batch_size = x.size(0)
        n_dim = x.ndim
        if n_dim == 4:
            n = x.size(1)
            assert x.size(1) == x.size(2)
        elif n_dim == 3:
            n = int(x.size(1)**(0.5))
        else:
            raise ValueError("Dimension not implemented")
        in_dim = self.in_dim
        out_dim = self.out_dim
        modes = self.modes

        x = x.view(-1, n, n, in_dim)
        res = self.linear(x)
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)
        x_ft = fft.rfft2(x, s=(n, n), norm=self.norm)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = torch.zeros(batch_size, out_dim, n, n //
                             2+1, 2, device=x.device)
        out_ft[:, :, :modes, :modes] = self.complex_matmul_2d(
            x_ft[:, :, :modes, :modes], self.fourier_weight[0])
        out_ft[:, :, -modes:, :modes] = self.complex_matmul_2d(
            x_ft[:, :, -modes:, :modes], self.fourier_weight[1])
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft2(out_ft, s=(n, n), norm=self.norm)
        x = x.permute(0, 2, 3, 1)
        x = self.activation(x + res)

        if n_dim == 3:
            x = x.view(batch_size, n**2, out_dim)

        if self.return_freq:
            return x, out_ft
        else:
            return x