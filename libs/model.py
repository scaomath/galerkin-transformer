try:
    from libs.layers import *
    from libs.utils_ft import *
except:
    from layers import *
    from utils_ft import *

import copy
import os
import sys
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MultiheadAttention, TransformerEncoderLayer
from torch.nn.init import constant_, xavier_uniform_
from torchinfo import summary

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
sys.path.append(SRC_ROOT)

ADDITIONAL_ATTR = ['normalizer', 'raw_laplacian', 'return_latent',
                   'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
                   'upscaler_size', 'downscaler_size', 'spacial_dim', 'spacial_fc',
                   'regressor_activation', 'attn_activation', 
                   'downscaler_activation', 'upscaler_activation',
                   'encoder_dropout', 'decoder_dropout', 'ffn_dropout']


class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float=1e-2,
                 diagonal_weight: float=1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False,
                 ):
        super(SimpleTransformerEncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        self.attn = SimpleAttention(n_head=n_head,
                                    d_model=d_model,
                                    attention_type=attention_type,
                                    diagonal_weight=diagonal_weight,
                                    xavier_init=xavier_init,
                                    symmetric_init=symmetric_init,
                                    pos_dim=pos_dim,
                                    norm=attn_norm,
                                    norm_type=norm_type,
                                    eps=norm_eps,
                                    dropout=dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional 
            information if coords are in features
        '''
        if self.add_pos_emb:
            x = x.permute((1, 0, 2))
            x = self.pos_emb(x)
            x = x.permute((1, 0, 2))

        if pos is not None and self.pos_dim > 0:
            att_output, attn_weight = self.attn(
                x, x, x, pos=pos, weight=weight)  # encoder no mask
        else:
            att_output, attn_weight = self.attn(x, x, x, weight=weight)

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x

class GalerkinTransformerDecoderLayer(nn.Module):
    r"""
    A lite implementation of the decoder layer based on linear causal attention
    adapted from the TransformerDecoderLayer in PyTorch
    https://github.com/pytorch/pytorch/blob/afc1d1b3d6dad5f9f56b1a4cb335de109adb6018/torch/nn/modules/transformer.py#L359
    """
    def __init__(self, d_model, 
                        nhead,
                        pos_dim = 1,
                        dim_feedforward=512, 
                        attention_type='galerkin',
                        layer_norm=True,
                        attn_norm=None,
                        norm_type='layer',
                        norm_eps=1e-5,
                        xavier_init: float=1e-2,
                        diagonal_weight: float = 1e-2,
                        dropout=0.05, 
                        ffn_dropout=None,
                        activation_type='relu',
                        device=None, 
                        dtype=None,
                        debug=False,) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype, }
        super(GalerkinTransformerDecoderLayer, self).__init__()

        ffn_dropout = default(ffn_dropout, dropout)
        self.debug = debug
        self.self_attn = SimpleAttention(nhead, d_model, 
                                        attention_type=attention_type,
                                        pos_dim=pos_dim,
                                        norm=attn_norm,
                                        eps=norm_eps,
                                        norm_type=norm_type,
                                        diagonal_weight=diagonal_weight,
                                        xavier_init=xavier_init,
                                        dropout=dropout,)
        self.multihead_attn = SimpleAttention(nhead, d_model, 
                                        attention_type='causal',
                                        pos_dim=pos_dim,
                                        norm=attn_norm,
                                        eps=norm_eps,
                                        norm_type=norm_type,
                                        diagonal_weight=diagonal_weight,
                                        xavier_init=xavier_init,
                                        dropout=dropout,)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.add_layer_norm = layer_norm
        if self.add_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        if self.add_layer_norm:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))
        else:
            x = x + self._sa_block(x, tgt_mask)
            x = x + self._mha_block(x, memory, memory_mask)
            x = x + self._ff_block(x)
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask,)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem, mask=attn_mask,)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.ff(x)
        return self.dropout(x)


class _TransformerEncoderLayer(nn.Module):
    r"""
    Taken from official torch implementation:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        - add a layer norm switch
        - add an attn_weight output switch
        - batch first
        batch_first has been added in PyTorch 1.9.0
        https://github.com/pytorch/pytorch/pull/55285
    """

    def __init__(self, d_model, nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm=True,
                 attn_weight=False,
                 ):
        super(_TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.add_layer_norm = layer_norm
        self.attn_weight = attn_weight
        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor,
                pos: Optional[Tensor] = None,
                weight: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args (modified from torch):
            src: the sequence to the encoder layer (required):  (batch_size, seq_len, d_model)
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.

        Remark: 
            PyTorch official implementation: (seq_len, n_batch, d_model) as input
            here we permute the first two dims as input
            so in the first line the dim needs to be permuted then permuted back
        """
        if pos is not None:
            src = torch.cat([pos, src], dim=-1)

        src = src.permute(1, 0, 2)

        if (src_mask is None) or (src_key_padding_mask is None):
            src2, attn_weight = self.self_attn(src, src, src)
        else:
            src2, attn_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                                               key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        if self.add_layer_norm:
            src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if self.add_layer_norm:
            src = self.norm2(src)
        src = src.permute(1, 0, 2)
        if self.attn_weight:
            return src, attn_weight
        else:
            return src


class TransformerEncoderWrapper(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
        Modified from pytorch official implementation
        TransformerEncoder's input and output shapes follow
        those of the encoder_layer fed into as this is essentially a wrapper

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers,
                 norm=None,):
        super(TransformerEncoderWrapper, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output


class GCN(nn.Module):
    def __init__(self,
                 node_feats=4,
                 out_features=96,
                 num_gcn_layers=2,
                 edge_feats=6,
                 activation=True,
                 raw_laplacian=False,
                 dropout=0.1,
                 debug=False):
        super(GCN, self).__init__()
        '''
        A simple GCN, a wrapper for Kipf and Weiling's code
        learnable edge features similar to 
        Graph Transformer https://arxiv.org/abs/1911.06455
        but using neighbor agg
        '''
        self.edge_learner = EdgeEncoder(out_dim=out_features,
                                        edge_feats=edge_feats,
                                        raw_laplacian=raw_laplacian
                                        )
        self.gcn_layer0 = GraphConvolution(in_features=node_feats,  # hard coded
                                           out_features=out_features,
                                           debug=debug,
                                           )
        self.gcn_layers = nn.ModuleList([copy.deepcopy(GraphConvolution(
            in_features=out_features,  # hard coded
            out_features=out_features,
            debug=debug
        )) for _ in range(1, num_gcn_layers)])
        self.activation = activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.edge_feats = edge_feats
        self.debug = debug

    def forward(self, x, edge):
        x = x.permute(0, 2, 1).contiguous()
        edge = edge.permute([0, 3, 1, 2]).contiguous()
        assert edge.size(1) == self.edge_feats

        edge = self.edge_learner(edge)

        out = self.gcn_layer0(x, edge)
        for gc in self.gcn_layers[:-1]:
            out = gc(out, edge)
            if self.activation:
                out = self.relu(out)

        # last layer no activation
        out = self.gcn_layers[-1](out, edge)
        return out.permute(0, 2, 1)


class GAT(nn.Module):
    def __init__(self,
                 node_feats=4,
                 out_features=96,
                 num_gcn_layers=2,
                 edge_feats=None,
                 activation=False,
                 debug=False):
        super(GAT, self).__init__()
        '''
        A simple GAT: modified from the official implementation
        '''
        self.gat_layer0 = GraphAttention(in_features=node_feats,
                                         out_features=out_features,
                                         )
        self.gat_layers = nn.ModuleList([copy.deepcopy(GraphAttention(
            in_features=out_features,
            out_features=out_features,
        )) for _ in range(1, num_gcn_layers)])
        self.activation = activation
        self.relu = nn.ReLU()
        self.debug = debug

    def forward(self, x, edge):
        '''
        input: node feats (-1, seq_len, n_feats)
               edge only takes adj (-1, seq_len, seq_len)
               edge matrix first one in the last dim is graph Lap.
        '''
        edge = edge[..., 0].contiguous()

        out = self.gat_layer0(x, edge)

        for layer in self.gat_layers[:-1]:
            out = layer(out, edge)
            if self.activation:
                out = self.relu(out)

        # last layer no activation
        return self.gat_layers[-1](out, edge)


class PointwiseRegressor(nn.Module):
    def __init__(self, in_dim,  # input dimension
                 n_hidden,
                 out_dim,  # number of target dim
                 num_layers: int = 2,
                 spacial_fc: bool = False,
                 spacial_dim=1,
                 dropout=0.1,
                 activation='silu',
                 return_latent=False,
                 debug=False):
        super(PointwiseRegressor, self).__init__()
        '''
        A wrapper for a simple pointwise linear layers
        '''
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc
        activ = nn.SiLU() if activation == 'silu' else nn.ReLU()
        if self.spacial_fc:
            in_dim = in_dim + spacial_dim
            self.fc = nn.Linear(in_dim, n_hidden)
        self.ff = nn.ModuleList([nn.Sequential(
                                nn.Linear(n_hidden, n_hidden),
                                activ,
                                )])
        for _ in range(num_layers - 1):
            self.ff.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activ,
            ))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_hidden, out_dim)
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.ff:
            x = layer(x)
            x = self.dropout(x)

        x = self.out(x)

        if self.return_latent:
            return x, None
        else:
            return x


class SpectralRegressor(nn.Module):
    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=False,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 dropout=0.1,
                 debug=False):
        super(SpectralRegressor, self).__init__()
        '''
        A wrapper for both SpectralConv1d and SpectralConv2d
        Ref: Li et 2020 FNO paper
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        A new implementation incoporating all spacial-based FNO
        in_dim: input dimension, (either n_hidden or spacial dim)
        n_hidden: number of hidden features out from attention to the fourier conv
        '''
        if spacial_dim == 2:  # 2d, function + (x,y)
            spectral_conv = SpectralConv2d
        elif spacial_dim == 1:  # 1d, function + x
            spectral_conv = SpectralConv1d
        else:
            raise NotImplementedError("3D not implemented.")
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
                                                          out_dim=freq_dim,
                                                          n_grid=n_grid,
                                                          modes=modes,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    n_grid=n_grid,
                                                    modes=modes,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = default(dim_feedforward, 2*spacial_dim*freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, edge=None, pos=None, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        else:
            return x


class DownScaler(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 dropout=0.1,
                 padding=5,
                 downsample_mode='conv',
                 activation_type='silu',
                 interp_size=None,
                 debug=False):
        super(DownScaler, self).__init__()
        '''
        A wrapper for conv2d/interp downscaler
        '''
        if downsample_mode == 'conv':
            self.downsample = nn.Sequential(Conv2dEncoder(in_dim=in_dim,
                                                          out_dim=out_dim,
                                                          activation_type=activation_type,
                                                          debug=debug),
                                            Conv2dEncoder(in_dim=out_dim,
                                                          out_dim=out_dim,
                                                          padding=padding,
                                                          activation_type=activation_type,
                                                          debug=debug))
        elif downsample_mode == 'interp':
            self.downsample = Interp2dEncoder(in_dim=in_dim,
                                              out_dim=out_dim,
                                              interp_size=interp_size,
                                              activation_type=activation_type,
                                              dropout=dropout,
                                              debug=debug)
        else:
            raise NotImplementedError("downsample mode not implemented.")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n, n, in_dim)
            Output: (-1, n_s, n_s, out_dim)
        '''
        n_grid = x.size(1)
        bsz = x.size(0)
        x = x.view(bsz, n_grid, n_grid, self.in_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)
        return x


class UpScaler(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dim=None,
                 padding=2,
                 output_padding=0,
                 dropout=0.1,
                 upsample_mode='conv',
                 activation_type='silu',
                 interp_mode='bilinear',
                 interp_size=None,
                 debug=False):
        super(UpScaler, self).__init__()
        '''
        A wrapper for DeConv2d upscaler or interpolation upscaler
        Deconv: Conv1dTranspose
        Interp: interp->conv->interp
        '''
        hidden_dim = default(hidden_dim, in_dim)
        if upsample_mode in ['conv', 'deconv']:
            self.upsample = nn.Sequential(
                DeConv2dBlock(in_dim=in_dim,
                              out_dim=out_dim,
                              hidden_dim=hidden_dim,
                              padding=padding,
                              output_padding=output_padding,
                              dropout=dropout,
                              activation_type=activation_type,
                              debug=debug),
                DeConv2dBlock(in_dim=in_dim,
                              out_dim=out_dim,
                              hidden_dim=hidden_dim,
                              padding=padding*2,
                              output_padding=output_padding,
                              dropout=dropout,
                              activation_type=activation_type,
                              debug=debug))
        elif upsample_mode == 'interp':
            self.upsample = Interp2dUpsample(in_dim=in_dim,
                                             out_dim=out_dim,
                                             interp_mode=interp_mode,
                                             interp_size=interp_size,
                                             dropout=dropout,
                                             activation_type=activation_type,
                                             debug=debug)
        else:
            raise NotImplementedError("upsample mode not implemented.")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n_s, n_s, in_dim)
            Output: (-1, n, n, out_dim)
        '''
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleTransformer, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer'

    def forward(self, node, edge, pos, grid=None, weight=None):
        '''
        seq_len: n, number of grid points
        node_feats: number of features of the inputs
        edge_feats: number of Laplacian matrices (including learned)
        pos_dim: dimension of the Euclidean space
        - node: (batch_size, seq_len, node_feats)
        - pos: (batch_size, seq_len, pos_dim)
        - edge: (batch_size, seq_len, seq_len, edge_feats)
        - weight: (batch_size, seq_len, seq_len): mass matrix prefered
            or (batch_size, seq_len) when mass matrices are not provided
        
        Remark:
        for classic Transformer: pos_dim = n_hidden = 512
        pos encodings is added to the latent representation
        '''
        x_latent = []
        attn_weights = []

        x = self.feat_extract(node, edge)

        if self.spacial_residual or self.return_latent:
            res = x.contiguous()
            x_latent.append(res)

        for encoder in self.encoder_layers:
            if self.return_attn_weight:
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            else:
                x = encoder(x, pos, weight)

            if self.return_latent:
                x_latent.append(x.contiguous())

        if self.spacial_residual:
            x = res + x

        x_freq = self.freq_regressor(
            x)[:, :self.pred_len, :] if self.n_freq_targets > 0 else None

        x = self.dpo(x)
        x = self.regressor(x, grid=grid)

        return dict(preds=x,
                    preds_freq=x_freq,
                    preds_latent=x_latent,
                    attn_weights=attn_weights)

    def _initialize(self):
        self._get_feature()

        self._get_encoder()

        if self.n_freq_targets > 0:
            self._get_freq_regressor()

        self._get_regressor()

        if self.decoder_type in ['pointwise', 'convolution']:
            self._initialize_layer(self.regressor)

        self.config = dict(self.config)

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.spacial_dim = default(self.spacial_dim, self.pos_dim)
        self.spacial_fc = default(self.spacial_fc, False)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        if self.num_feat_layers > 0 and self.feat_extract_type == 'gcn':
            self.feat_extract = GCN(node_feats=self.node_feats,
                                    edge_feats=self.edge_feats,
                                    num_gcn_layers=self.num_feat_layers,
                                    out_features=self.n_hidden,
                                    activation=self.graph_activation,
                                    raw_laplacian=self.raw_laplacian,
                                    debug=self.debug,
                                    )
        elif self.num_feat_layers > 0 and self.feat_extract_type == 'gat':
            self.feat_extract = GAT(node_feats=self.node_feats,
                                    out_features=self.n_hidden,
                                    num_gcn_layers=self.num_feat_layers,
                                    activation=self.graph_activation,
                                    debug=self.debug,
                                    )
        else:
            self.feat_extract = Identity(in_features=self.node_feats,
                                         out_features=self.n_hidden)

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                           n_head=self.n_head,
                                                           attention_type=self.attention_type,
                                                           dim_feedforward=self.dim_feedforward,
                                                           layer_norm=self.layer_norm,
                                                           attn_norm=self.attn_norm,
                                                           norm_type=self.norm_type,
                                                           batch_norm=self.batch_norm,
                                                           pos_dim=self.pos_dim,
                                                           xavier_init=self.xavier_init,
                                                           diagonal_weight=self.diagonal_weight,
                                                           symmetric_init=self.symmetric_init,
                                                           attn_weight=self.return_attn_weight,
                                                           residual_type=self.residual_type,
                                                           activation_type=self.attn_activation,
                                                           dropout=self.encoder_dropout,
                                                           ffn_dropout=self.ffn_dropout,
                                                           debug=self.debug)
        else:
            encoder_layer = _TransformerEncoderLayer(d_model=self.n_hidden,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    layer_norm=self.layer_norm,
                                                    attn_weight=self.return_attn_weight,
                                                    dropout=self.encoder_dropout
                                                    )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_freq_regressor(self):
        if self.bulk_regression:
            self.freq_regressor = BulkRegressor(in_dim=self.seq_len,
                                                n_feats=self.n_hidden,
                                                n_targets=self.n_freq_targets,
                                                pred_len=self.pred_len)
        else:
            self.freq_regressor = nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_freq_targets),
            )

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.n_targets,
                                                spacial_fc=self.spacial_fc,
                                                spacial_dim=self.spacial_dim,
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                debug=self.debug)
        elif self.decoder_type == 'ifft':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.n_hidden,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.n_targets,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               dim_feedforward=self.freq_dim,
                                               activation=self.regressor_activation,
                                               dropout=self.decoder_dropout,
                                               )
        else:
            raise NotImplementedError("Decoder type not implemented")

    def get_graph(self):
        return self.gragh

    def get_encoder(self):
        return self.encoder_layers


class FourierTransformer2D(nn.Module):
    def __init__(self, **kwargs):
        super(FourierTransformer2D, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer2D'

    def forward(self, node, edge, pos, grid, weight=None, boundary_value=None):
        '''
        - node: (batch_size, n, n, node_feats)
        - pos: (batch_size, n_s*n_s, pos_dim)
        - edge: (batch_size, n_s*n_s, n_s*n_s, edge_feats)
        - weight: (batch_size, n_s*n_s, n_s*n_s): mass matrix prefered
            or (batch_size, n_s*n_s) when mass matrices are not provided (lumped mass)
        - grid: (batch_size, n-2, n-2, 2) excluding boundary
        '''
        bsz = node.size(0)
        n_s = int(pos.size(1)**(0.5))
        x_latent = []
        attn_weights = []

        if not self.downscaler_size:
            node = torch.cat(
                [node, pos.contiguous().view(bsz, n_s, n_s, -1)], dim=-1)
        x = self.downscaler(node)
        x = x.view(bsz, -1, self.n_hidden)

        x = self.feat_extract(x, edge)
        x = self.dpo(x)

        for encoder in self.encoder_layers:
            if self.return_attn_weight and self.attention_type != 'official':
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            elif self.attention_type != 'official':
                x = encoder(x, pos, weight)
            else:
                out_dim = self.n_head*self.pos_dim + self.n_hidden
                x = x.view(bsz, -1, self.n_head, self.n_hidden//self.n_head).transpose(1, 2)
                x = torch.cat([pos.repeat([1, self.n_head, 1, 1]), x], dim=-1)
                x = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)
                x = encoder(x)
            if self.return_latent:
                x_latent.append(x.contiguous())

        x = x.view(bsz, n_s, n_s, self.n_hidden)
        x = self.upscaler(x)

        if self.return_latent:
            x_latent.append(x.contiguous())

        x = self.dpo(x)

        if self.return_latent:
            x, xr_latent = self.regressor(x, grid=grid)
            x_latent.append(xr_latent)
        else:
            x = self.regressor(x, grid=grid)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.boundary_condition == 'dirichlet':
            x = x[:, 1:-1, 1:-1].contiguous()
            x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)
            if boundary_value is not None:
                assert x.size() == boundary_value.size()
                x += boundary_value

        return dict(preds=x,
                    preds_latent=x_latent,
                    attn_weights=attn_weights)

    def _initialize(self):
        self._get_feature()
        self._get_scaler()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def cuda(self, device=None):
        self = super().cuda(device)
        if self.normalizer:
            self.normalizer = self.normalizer.cuda(device)
        return self

    def cpu(self):
        self = super().cpu()
        if self.normalizer:
            self.normalizer = self.normalizer.cpu()
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if self.normalizer:
            self.normalizer = self.normalizer.to(*args, **kwargs)
        return self

    def print_config(self):
        for a in self.config.keys():
            if not a.startswith('__'):
                print(f"{a}: \t", getattr(self, a))

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    @staticmethod
    def _get_pos(pos, downsample):
        '''
        get the downscaled position in 2d
        '''
        bsz = pos.size(0)
        n_grid = pos.size(1)
        x, y = pos[..., 0], pos[..., 1]
        x = x.view(bsz, n_grid, n_grid)
        y = y.view(bsz, n_grid, n_grid)
        x = x[:, ::downsample, ::downsample].contiguous()
        y = y[:, ::downsample, ::downsample].contiguous()
        return torch.stack([x, y], dim=-1)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral', 'local', 'global',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        if self.feat_extract_type == 'gcn' and self.num_feat_layers > 0:
            self.feat_extract = GCN(node_feats=self.n_hidden,
                                    edge_feats=self.edge_feats,
                                    num_gcn_layers=self.num_feat_layers,
                                    out_features=self.n_hidden,
                                    activation=self.graph_activation,
                                    raw_laplacian=self.raw_laplacian,
                                    debug=self.debug,
                                    )
        elif self.feat_extract_type == 'gat' and self.num_feat_layers > 0:
            self.feat_extract = GAT(node_feats=self.n_hidden,
                                    out_features=self.n_hidden,
                                    num_gcn_layers=self.num_feat_layers,
                                    activation=self.graph_activation,
                                    debug=self.debug,
                                    )
        else:
            self.feat_extract = Identity()

    def _get_scaler(self):
        if self.downscaler_size:
            self.downscaler = DownScaler(in_dim=self.node_feats,
                                         out_dim=self.n_hidden,
                                         downsample_mode=self.downsample_mode,
                                         interp_size=self.downscaler_size,
                                         dropout=self.downscaler_dropout,
                                         activation_type=self.downscaler_activation)
        else:
            self.downscaler = Identity(in_features=self.node_feats+self.spacial_dim,
                                       out_features=self.n_hidden)
        if self.upscaler_size:
            self.upscaler = UpScaler(in_dim=self.n_hidden,
                                     out_dim=self.n_hidden,
                                     upsample_mode=self.upsample_mode,
                                     interp_size=self.upscaler_size,
                                     dropout=self.upscaler_dropout,
                                     activation_type=self.upscaler_activation)
        else:
            self.upscaler = Identity()

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                           n_head=self.n_head,
                                                           attention_type=self.attention_type,
                                                           dim_feedforward=self.dim_feedforward,
                                                           layer_norm=self.layer_norm,
                                                           attn_norm=self.attn_norm,
                                                           batch_norm=self.batch_norm,
                                                           pos_dim=self.pos_dim,
                                                           xavier_init=self.xavier_init,
                                                           diagonal_weight=self.diagonal_weight,
                                                           symmetric_init=self.symmetric_init,
                                                           attn_weight=self.return_attn_weight,
                                                           dropout=self.encoder_dropout,
                                                           ffn_dropout=self.ffn_dropout,
                                                           norm_eps=self.norm_eps,
                                                           debug=self.debug)
        elif self.attention_type == 'official':
            encoder_layer = TransformerEncoderLayer(d_model=self.n_hidden+self.pos_dim*self.n_head,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=self.encoder_dropout,
                                                    batch_first=True,
                                                    layer_norm_eps=self.norm_eps,
                                                    )
        else:
            raise NotImplementedError("encoder type not implemented.")
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.n_targets,
                                                num_layers=self.num_regressor_layers,
                                                spacial_fc=self.spacial_fc,
                                                spacial_dim=self.spacial_dim,
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                return_latent=self.return_latent,
                                                debug=self.debug)
        elif self.decoder_type == 'ifft2':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.freq_dim,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.n_targets,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               activation=self.regressor_activation,
                                               last_activation=self.last_activation,
                                               dropout=self.decoder_dropout,
                                               return_latent=self.return_latent,
                                               debug=self.debug
                                               )
        else:
            raise NotImplementedError("Decoder type not implemented")

class FourierTransformer2DLite(nn.Module):
    '''
    A lite model of the Fourier/Galerkin Transformer
    '''
    def __init__(self, **kwargs):
        super(FourierTransformer2DLite, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()

    def forward(self, node, edge, pos, grid=None):
        '''
        seq_len: n, number of grid points
        node_feats: number of features of the inputs
        pos_dim: dimension of the Euclidean space
        - node: (batch_size, n*n, node_feats)
        - pos: (batch_size, n*n, pos_dim)
        - grid: (batch_size, n, n, pos_dim)

        Remark:
        for classic Transformer: pos_dim = n_hidden = 512
        pos encodings is added to the latent representation
        '''
        bsz = node.size(0)
        input_dim = node.size(-1)
        n_grid = grid.size(1)
        node = torch.cat([node.view(bsz, -1, input_dim), pos],
                         dim=-1)
        x = self.feat_extract(node, edge)

        for encoder in self.encoder_layers:
            x = encoder(x, pos)

        x = self.dpo(x)
        x = x.view(bsz, n_grid, n_grid, -1)
        x = self.regressor(x, grid=grid)

        return dict(preds=x,
                    preds_freq=None,
                    preds_latent=None,
                    attn_weights=None)

    def _initialize(self):
        self._get_feature()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.spacial_dim = default(self.spacial_dim, self.pos_dim)
        self.spacial_fc = default(self.spacial_fc, False)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        self.feat_extract = Identity(in_features=self.node_feats,
                                     out_features=self.n_hidden)

    def _get_encoder(self):
        encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                       n_head=self.n_head,
                                                       dim_feedforward=self.dim_feedforward,
                                                       layer_norm=self.layer_norm,
                                                       attention_type=self.attention_type,
                                                       attn_norm=self.attn_norm,
                                                       norm_type=self.norm_type,
                                                       xavier_init=self.xavier_init,
                                                       diagonal_weight=self.diagonal_weight,
                                                       dropout=self.encoder_dropout,
                                                       ffn_dropout=self.ffn_dropout,
                                                       pos_dim=self.pos_dim,
                                                       debug=self.debug)

        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                           n_hidden=self.n_hidden,
                                           freq_dim=self.freq_dim,
                                           out_dim=self.n_targets,
                                           num_spectral_layers=self.num_regressor_layers,
                                           modes=self.fourier_modes,
                                           spacial_dim=self.spacial_dim,
                                           spacial_fc=self.spacial_fc,
                                           dim_feedforward=self.freq_dim,
                                           activation=self.regressor_activation,
                                           dropout=self.decoder_dropout,
                                           )


if __name__ == '__main__':
    for graph in ['gcn', 'gat']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = defaultdict(lambda: None,
                            node_feats=1,
                            edge_feats=5,
                            pos_dim=1,
                            n_targets=1,
                            n_hidden=96,
                            num_feat_layers=2,
                            num_encoder_layers=2,
                            n_head=2,
                            pred_len=0,
                            n_freq_targets=0,
                            dim_feedforward=96*2,
                            feat_extract_type=graph,
                            graph_activation=True,
                            raw_laplacian=True,
                            attention_type='fourier',  # no softmax
                            xavier_init=1e-4,
                            diagonal_weight=1e-2,
                            symmetric_init=False,
                            layer_norm=True,
                            attn_norm=False,
                            batch_norm=False,
                            spacial_residual=False,
                            return_attn_weight=True,
                            seq_len=None,
                            bulk_regression=False,
                            decoder_type='ifft',
                            freq_dim=64,
                            num_regressor_layers=2,
                            fourier_modes=16,
                            spacial_dim=1,
                            spacial_fc=True,
                            dropout=0.1,
                            debug=False,
                            )

        ft = SimpleTransformer(**config)
        ft.to(device)
        batch_size, seq_len = 8, 512
        summary(ft, input_size=[(batch_size, seq_len, 1),
                                (batch_size, seq_len, seq_len, 5),
                                (batch_size, seq_len, 1),
                                (batch_size, seq_len, 1)], device=device)

    layer = TransformerEncoderLayer(d_model=128, nhead=4)
    print(layer.__class__)
