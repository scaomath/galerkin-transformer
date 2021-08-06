import argparse
import math
import os
import sys
import re
from collections import OrderedDict
from datetime import date

import numpy as np
import pandas as pd
import torch
from matplotlib import rc, rcParams, tri
from numpy.core.numeric import identity
from scipy.io import loadmat
from scipy.sparse import csr_matrix, diags
from scipy.sparse import hstack as sparse_hstack
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from libs.utils import *
except:
    from utils import *

try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError as e:
    print('Please install Plotly for showing mesh and solutions.')

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
MODEL_PATH = default(os.environ.get('MODEL_PATH'),
                     os.path.join(SRC_ROOT, 'models'))
DATA_PATH = default(os.environ.get('DATA_PATH'),
                    os.path.join(SRC_ROOT, 'data'))
FIG_PATH = default(os.environ.get('FIG_PATH'),
                   os.path.join(os.path.dirname(SRC_ROOT), 'figures'))
EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']
PI = math.pi
SEED = default(os.environ.get('SEED'), 1127802)


def clones(module, N):
    '''
    Input:
        - module: nn.Module obj
    Output:
        - zip identical N layers (not stacking)

    Refs:
        - https://nlp.seas.harvard.edu/2018/04/03/attention.html
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def csr_to_sparse(M):
    '''    
    Input: 
        M: csr_matrix
    Output:
        torch sparse tensor

    Another implementation can be found in
    https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    '''
    n, m = M.shape
    coo_ = M.tocoo()
    ix = torch.LongTensor([coo_.row, coo_.col])
    M_t = torch.sparse.FloatTensor(ix,
                                   torch.from_numpy(M.data).float(),
                                   [n, m])
    return M_t


def pooling_2d(mat, kernel_size: tuple = (2, 2), method='mean', padding=False):
    '''Non-overlapping pooling on 2D data (or 2D data stacked as 3D array).

    mat: ndarray, input array to pool. (m, n) or (bsz, m, n)
    kernel_size: tuple of 2, kernel size in (ky, kx).
    method: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    pad: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f), padding is nan
           so when computing mean the 0 is counted

    Return <result>: pooled matrix.

    Modified from https://stackoverflow.com/a/49317610/622119
    to handle the case of batch edge matrices
    CC BY-SA 3.0
    '''

    m, n = mat.shape[-2:]
    ky, kx = kernel_size

    def _ceil(x, y): return int(np.ceil(x/float(y)))

    if padding:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = mat.shape[:-2] + (ny*ky, nx*kx)
        sy = (ny*ky - m)//2
        sx = (nx*kx - n)//2
        _sy = ny*ky - m - sy
        _sx = nx*kx - n - sx

        mat_pad = np.full(size, np.nan)
        mat_pad[..., sy:-_sy, sx:-_sx] = mat
    else:
        ny = m//ky
        nx = n//kx
        mat_pad = mat[..., :ny*ky, :nx*kx]

    new_shape = mat.shape[:-2] + (ny, ky, nx, kx)

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(-3, -1))
    elif method == 'mean':
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(-3, -1))
    else:
        raise NotImplementedError("pooling method not implemented.")

    return result


def quadpts(order=2):
    '''
    ported from Long Chen's iFEM's quadpts
    '''

    if order == 1:     # Order 1, nQuad 1
        baryCoords = [1/3, 1/3, 1/3]
        weight = 1
    elif order == 2:    # Order 2, nQuad 3
        baryCoords = [[2/3, 1/6, 1/6],
                      [1/6, 2/3, 1/6],
                      [1/6, 1/6, 2/3]]
        weight = [1/3, 1/3, 1/3]
    elif order == 3:     # Order 3, nQuad 4
        baryCoords = [[1/3, 1/3, 1/3],
                      [0.6, 0.2, 0.2],
                      [0.2, 0.6, 0.2],
                      [0.2, 0.2, 0.6]]
        weight = [-27/48, 25/48, 25/48, 25/48]
    elif order == 4:     # Order 4, nQuad 6
        baryCoords = [[0.108103018168070, 0.445948490915965, 0.445948490915965],
                      [0.445948490915965, 0.108103018168070, 0.445948490915965],
                      [0.445948490915965, 0.445948490915965, 0.108103018168070],
                      [0.816847572980459, 0.091576213509771, 0.091576213509771],
                      [0.091576213509771, 0.816847572980459, 0.091576213509771],
                      [0.091576213509771, 0.091576213509771, 0.816847572980459], ]
        weight = [0.223381589678011, 0.223381589678011, 0.223381589678011,
                  0.109951743655322, 0.109951743655322, 0.109951743655322]
    return np.array(baryCoords), np.array(weight)


def get_distance_matrix(node, graph=False):
    '''
    Input:
        - Node: nodal coords
        - graph: bool, whether to return graph distance
    Output:
        - inverse distance matrices (linear and square)
          (batch_size, N, N, 2)
    '''
    N = len(node)
    idx = np.arange(N)
    Ds = []
    for i in range(len(idx)):
        if graph:
            d = np.abs(idx[i] - idx)
        else:
            d = np.abs(node[i] - node)
        Ds.append(d)

    Dss = []
    if graph:
        Ds = np.array(Ds) + 1
        Ds = 1 / Ds
        Ds = np.repeat(Ds, 1, axis=0)
        for i in [1, 2]:
            Dss.append(Ds ** i)
    else:
        Ds = np.array(Ds)
        max_distance = Ds.max()
        Ds /= (max_distance + 1e-8)
        Dss.append(np.exp(-Ds))
        Ds = 1 / (1+Ds)
        Dss.append(Ds)

    Ds = np.stack(Dss, axis=2)

    return Ds


def get_laplacian_1d(node,
                     K=None,
                     weight=None,  # weight for renormalization
                     normalize=True,
                     smoother=None):
    '''
    Construct the 1D Laplacian matrix on the domain defined by node. 
    with a variable mesh size.

    Input:
        - node: array-like, shape (N, ) One dimensional mesh; or a positve integer.
        - normalize: apply D^{-1/2} A D^{-1/2} row and column scaling to the Laplacian 

    Output:
        - A : scipy sparse matrix, shape (N, N)
        Laplacian matrix.

    Reference:
        Code adapted to 1D from the 2D one in 
        Long Chen: iFEM: An innovative finite element method package in Matlab. 
        Technical report, University of California-Irvine, 2009
    '''
    if isinstance(node, int):
        node = np.linspace(0, 1, node)
    N = node.shape[0]
    h = node[1:] - node[:-1]
    elem = np.c_[np.arange(N-1), np.arange(1, N)]
    Dphi = np.c_[-1/h, 1/h]

    if K is None:
        K = 1

   # stiffness matrix
    A = csr_matrix((N, N))
    for i in range(2):
        for j in range(2):
            # $A_{ij}|_{\tau} = \int_{\tau}K\nabla \phi_i\cdot \nabla \phi_j dxdy$
            Aij = h*K*Dphi[:, i]*Dphi[:, j]
            A += csr_matrix((Aij, (elem[:, i], elem[:, j])), shape=(N, N))

    if weight is not None:
        A += diags(weight)

    if normalize:
        D = diags(A.diagonal()**(-0.5))
        A = (D.dot(A)).dot(D)

        if smoother == 'jacobi':
            I = identity(N)
            A = I-(2/3)*A  # jacobi
            A = csr_matrix(A)
        elif smoother == 'gs':
            raise NotImplementedError("Gauss-seidel not implemented")

    return A


def get_mass_1d(node, K=None, normalize=False):
    '''
    Construct the 1D Mass matrix on the domain defined by node. 
    with a variable mesh size.

    Input:
        - node: array-like, shape (N, ) One dimensional mesh.
        - normalize: apply D^{-1/2} M D^{-1/2} row and column scaling to the mass matrix 

    Output:
        - M : scipy sparse matrix, shape (N, N), mass matrix.

    Reference:
        Code adapted to 1D from the 2D one in 
        Long Chen: iFEM: An innovative finite element method package in Matlab. 
        Technical report, University of California-Irvine, 2009

    '''
    if isinstance(node, int):
        node = np.linspace(0, 1, node)
    N = node.shape[0]
    h = node[1:] - node[:-1]
    elem = np.c_[np.arange(N-1), np.arange(1, N)]
    if K is None:
        K = 1

   # mass matrix
    M = csr_matrix((N, N))
    for i in range(2):
        for j in range(2):
            # $M_{ij}|_{\tau} = \int_{\tau}K \phi_i \cdot \phi_j dx$ \
            Mij = h*K*((i == j)+1)/6
            M += csr_matrix((Mij, (elem[:, i], elem[:, j])), shape=(N, N))

    if normalize:
        D = diags(M.diagonal()**(-0.5))
        M = (D.dot(M)).dot(D)

    return M


def showmesh(node, elem, **kwargs):
    triangulation = tri.Triangulation(node[:, 0], node[:, 1], elem)
    markersize = 3000/len(node)
    if kwargs.items():
        h = plt.triplot(triangulation, 'b-h', **kwargs)
    else:
        h = plt.triplot(triangulation, 'b-h', linewidth=0.5,
                        alpha=0.5, markersize=markersize)
    return h


def showsolution(node, elem, u, **kwargs):
    '''
    show 2D solution either of a scalar function or a vector field
    on triangulations
    '''
    markersize = 3000/len(node)

    if u.ndim == 1:  # (N, )
        uplot = ff.create_trisurf(x=node[:, 0], y=node[:, 1], z=u,
                                  simplices=elem,
                                  colormap="Viridis",  # similar to matlab's default colormap
                                  showbackground=True,
                                  show_colorbar=False,
                                  aspectratio=dict(x=1, y=1, z=1),
                                  )
        fig = go.Figure(data=uplot)

    elif u.ndim == 2 and u.shape[1] == 2:  # (N, 2)
        if u.shape[0] == elem.shape[0]:
            u /= (np.abs(u)).max()
            node = node[elem].mean(axis=1)

        uplot = ff.create_quiver(x=node[:, 0], y=node[:, 1],
                                 u=u[:, 0], v=u[:, 1],
                                 scale=.2,
                                 arrow_scale=.5,
                                 name='gradient of u',
                                 line_width=1,
                                 )

        fig = go.Figure(data=uplot)

    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    fig.show()


def showsurf(x, y, z, **kwargs):
    '''
    show 2D solution either of a scalar function or a vector field
    on a meshgrid
    x, y, z: (M, N) matrix
    '''

    uplot = go.Surface(x=x, y=y, z=z,
                       colorscale='Viridis',
                       showscale=False),

    fig = go.Figure(data=uplot)

    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    fig.show()


def showcontour(z, **kwargs):
    '''
    show 2D solution z of its contour
    '''
    uplot = go.Contour(z=z,
                       colorscale='RdYlBu',
                       line_smoothing=0.85,
                       line_width=0.1,
                       contours=dict(
                           coloring='heatmap',
                           #    showlabels=True,
                       )
                       )
    fig = go.Figure(data=uplot,
                    layout={'xaxis': {'title': 'x-label',
                                      'visible': False,
                                      'showticklabels': False},
                            'yaxis': {'title': 'y-label',
                                      'visible': False,
                                      'showticklabels': False}},)
    fig.update_traces(showscale=False)
    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=0, r=0, t=0, b=0),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          **kwargs)
    fig.show()
    return fig


def showresult(result=dict(), title=None, result_type='convergence',
               u=None, uh=None, grid=None, elem=None):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    jtplot.style(theme='grade3', context='notebook', ticks=True, grid=False)

    if result_type == 'convergence':
        loss_train = result['loss_train']
        loss_val = result['loss_val']
        train_label = r"$\mathrm{Train}: {E}\left( \displaystyle\|u_f - u_h\|_{\alpha, h, 1} \right)$"
        plt.semilogy(loss_train, label=train_label)
        val_label = r'$\mathrm{Valid}: \|T_f - T_h\|_{-1, V_h}$'
        plt.semilogy(loss_val, label=val_label)
        plt.grid(True, which="both", ls="--")
        plt.legend(fontsize='x-large')
        if title == 'fourier':
            title_str = r"$\mathrm{Fourier}\ \mathrm{transformer}$"
        elif title == 'galerkin':
            title_str = r"$\mathrm{Galerkin}\ \mathrm{transformer}$"
        elif title == 'spectral':
            title_str = r"$\mathrm{Fourier}\ \mathrm{neural}\ \mathrm{operator}$"
        else:
            title_str = r"$\mathrm{Loss}\ \mathrm{result}$"
        plt.title(title_str, fontsize='xx-large')

    elif result_type == 'solutions':
        sample_len = len(u)
        i = np.random.choice(sample_len)
        u = u[i].cpu().numpy().reshape(-1)
        uh = uh[i].cpu().numpy().reshape(-1)
        showsolution(grid, elem, u, template='seaborn', width=600, height=500)
        showsolution(grid, elem, uh, template='seaborn',
                     width=600, height=500,)


def get_model_name(model='burgers',
                   num_encoder_layers=4,
                   n_hidden=96,
                   attention_type='fourier',
                   layer_norm=True,
                   grid_size=512,
                   inverse_problem=False,
                   additional_str: str = '',
                   ):

    model_name = 'burgers_' if model == 'burgers' else 'darcy_'
    if inverse_problem:
        model_name += 'inv_'
    model_name += str(grid_size)+'_'
    if attention_type == 'fourier':
        attn_str = f'{num_encoder_layers}ft_'
    elif attention_type == 'galerkin':
        attn_str = f'{num_encoder_layers}gt_'
    elif attention_type == 'linear':
        attn_str = f'{num_encoder_layers}lt_'
    elif attention_type == 'softmax':
        attn_str = f'{num_encoder_layers}st_'
    else:
        attn_str = f'{num_encoder_layers}att_'
    model_name += attn_str
    model_name += f'{n_hidden}d_'
    ln_str = 'ln_' if layer_norm else 'qkv_'
    model_name += ln_str
    if additional_str:
        model_name += additional_str

    _suffix = str(date.today())
    if model_name[-1] == '_':
        result_name = model_name + _suffix + '.pkl'
        model_name += _suffix + '.pt'
    else:
        result_name = model_name + '_' + _suffix + '.pkl'
        model_name += '_' + _suffix + '.pt'
    return model_name, result_name


def get_args_1d():
    parser = argparse.ArgumentParser(description='Example 1: Burgers equation')
    parser.add_argument('--subsample', type=int, default=4, metavar='subsample',
                        help='input sampling from 8192 (default: 4 i.e., 2048 grid)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='bsz',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--val-batch-size', type=int, default=4, metavar='bsz',
                        help='input batch size for validation (default: 4)')
    parser.add_argument('--attention-type', type=str, default='fourier', metavar='attn_type',
                        help='input attention type for encoders (possile: fourier (alias integral, local), galerkin (alias global), softmax (official PyTorch implementation), linear (standard Q(K^TV) with softmax), default: fourier)')
    parser.add_argument('--xavier-init', type=float, default=0.01, metavar='xavier_init',
                        help='input Xavier initialization strength for Q,K,V weights (default: 0.01)')
    parser.add_argument('--diagonal-weight', type=float, default=0.01, metavar='diagonal weight',
                        help='input diagonal weight initialization strength for Q,K,V weights (default: 0.01)')
    parser.add_argument('--ffn-dropout', type=float, default=0.0, metavar='ffn_dropout',
                        help='dropout for the FFN in attention (default: 0.0)')
    parser.add_argument('--encoder-dropout', type=float, default=0.0, metavar='encoder_dropout',
                        help='dropout after the scaled dot-product in attention (default: 0.0)')
    parser.add_argument('--decoder-dropout', type=float, default=0.0, metavar='decoder_dropout',
                        help='dropout for the decoder layers (default: 0.0)')
    parser.add_argument('--layer-norm', action='store_true', default=False,
                        help='use the conventional layer normalization')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='max learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='regularizer',
                        help='strength of gradient regularizer (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--show-batch', action='store_true', default=False,
                        help='show batch training result')
    parser.add_argument('--seed', type=int, default=SEED, metavar='Seed',
                        help='random seed (default: 1127802)')
    return parser.parse_args()


def get_args_2d(subsample_nodes=3,
                subsample_attn=10,
                gamma=0.5,
                noise=0.0,
                ffn_dropout=0.1,
                encoder_dropout=0.05,
                decoder_dropout=0.0,
                dropout=0.0,
                inverse=False,
                **kwargs):
    if inverse:
        parser = argparse.ArgumentParser(
            description='Example 3: inverse coefficient identification problem for Darcy interface flow')
    else:
        parser = argparse.ArgumentParser(
            description='Example 2: Darcy interface flow')

    n_grid = int(((421 - 1)/subsample_nodes) + 1)
    n_grid_c = int(((421 - 1)/subsample_attn) + 1)

    parser.add_argument('--subsample-nodes', type=int, default=subsample_nodes, metavar='subsample',
                        help=f'input fine grid sampling from 421x421 (default: {subsample_nodes} i.e., {n_grid}x{n_grid} grid)')
    parser.add_argument('--subsample-attn', type=int, default=6, metavar='subsample_attn',
                        help=f'input coarse grid sampling from 421x421 (default: {subsample_attn} i.e., {n_grid_c}x{n_grid_c} grid)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='bsz',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--val-batch-size', type=int, default=4, metavar='bsz',
                        help='input batch size for validation (default: 4)')
    parser.add_argument('--attention-type', type=str, default='galerkin', metavar='attn_type',
                        help='input attention type for encoders (possile: fourier (alias integral, local), galerkin (alias global), softmax (official PyTorch implementation), linear (standard Q(K^TV) with softmax), default: galerkin)')
    parser.add_argument('--noise', type=float, default=noise, metavar='noise',
                        help=f'strength of noise imposed (default: {noise})')
    parser.add_argument('--xavier-init', type=float, default=1e-2, metavar='xavier_init',
                        help='input Xavier initialization strength for Q,K,V weights (default: 0.01)')
    parser.add_argument('--diagonal-weight', type=float, default=1e-2, metavar='diagonal weight',
                        help='input diagonal weight initialization strength for Q,K,V weights (default: 0.01)')
    parser.add_argument('--ffn-dropout', type=float, default=ffn_dropout, metavar='ffn_dropout',
                        help=f'dropout for the FFN in attention (default: {ffn_dropout})')
    parser.add_argument('--encoder-dropout', type=float, default=encoder_dropout, metavar='encoder_dropout',
                        help=f'dropout after the scaled dot-product in attention (default: {encoder_dropout})')
    parser.add_argument('--dropout', type=float, default=dropout, metavar='dropout',
                        help=f'dropout before the decoder layers (default: {dropout})')
    parser.add_argument('--decoder-dropout', type=float, default=decoder_dropout, metavar='decoder_dropout',
                        help=f'dropout in the decoder layers (default: {decoder_dropout})')
    parser.add_argument('--layer-norm', action='store_true', default=False,
                        help='use the conventional layer normalization')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='max learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=gamma, metavar='regularizer',
                        help=f'strength of gradient regularizer (default: {gamma})')
    parser.add_argument('--no-scale-factor', action='store_true', default=False,
                        help='use size instead of scale factor in interpolation')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--show-batch', action='store_true', default=False,
                        help='show batch training result')
    parser.add_argument('--seed', type=int, default=SEED, metavar='Seed',
                        help='random seed (default: 1127802)')
    return parser.parse_args()


def train_batch_burgers(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.999):
    optimizer.zero_grad()
    x, edge = data["node"].to(device), data["edge"].to(device)
    pos, grid = data['pos'].to(device), data['grid'].to(device)
    out_ = model(x, edge, pos, grid)

    if isinstance(out_, dict):
        out = out_['preds']
        y_latent = out_['preds_latent']
    elif isinstance(out_, tuple):
        out = out_[0]
        y_latent = None

    target = data["target"].to(device)
    u, up = target[..., 0], target[..., 1]

    if out.size(2) == 2:
        u_pred, up_pred = out[..., 0], out[..., 1]
        loss, reg, ortho, _ = loss_func(
            u_pred, u, up_pred, up, preds_latent=y_latent)
    elif out.size(2) == 1:
        u_pred = out[..., 0]
        loss, reg, ortho, _ = loss_func(
            u_pred, u, targets_prime=up, preds_latent=y_latent)
    loss = loss + reg + ortho
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()
    try:
        up_pred = out[..., 1]
    except:
        up_pred = u_pred

    return (loss.item(), reg.item(), ortho.item()), u_pred, up_pred


def validate_epoch_burgers(model, metric_func, valid_loader, device):
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            x, edge = data["node"].to(device), data["edge"].to(device)
            pos, grid = data['pos'].to(device), data['grid'].to(device)
            out_ = model(x, edge, pos, grid)

            if isinstance(out_, dict):
                u_pred = out_['preds'][..., 0]
            elif isinstance(out_, tuple):
                u_pred = out_[0][..., 0]

            target = data["target"].to(device)
            u = target[..., 0]
            _, _, _, metric = metric_func(u_pred, u)
            try:
                metric_val.append(metric.item())
            except:
                metric_val.append(metric)

    return dict(metric=np.mean(metric_val, axis=0))


def train_batch_darcy(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99):
    optimizer.zero_grad()
    a, x, edge = data["coeff"].to(device), data["node"].to(
        device), data["edge"].to(device)
    pos, grid = data['pos'].to(device), data['grid'].to(device)
    u, gradu = data["target"].to(device), data["target_grad"].to(device)

    # pos is for attention, grid is the finest grid
    out_ = model(x, edge, pos=pos, grid=grid)
    if isinstance(out_, dict):
        out = out_['preds']
    elif isinstance(out_, tuple):
        out = out_[0]

    if out.ndim == 4:
        u_pred, pred_grad, target = out[..., 0], out[..., 1:], u[..., 0]
        loss, reg, _, _ = loss_func(u_pred, target, pred_grad, gradu, K=a)
    elif out.ndim == 3:
        u_pred, u = out[..., 0], u[..., 0]
        loss, reg, _, _ = loss_func(u_pred, u, targets_prime=gradu, K=a)
    loss = loss + reg
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()
    try:
        up_pred = out[..., 1:]
    except:
        up_pred = u_pred

    return (loss.item(), reg.item()), u_pred, up_pred


def validate_epoch_darcy(model, metric_func, valid_loader, device):
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            x, edge = data["node"].to(device), data["edge"].to(device)
            pos, grid = data['pos'].to(device), data['grid'].to(device)
            out_ = model(x, edge, pos=pos, grid=grid)
            if isinstance(out_, dict):
                out = out_['preds']
            elif isinstance(out_, tuple):
                out = out_[0]
            u_pred = out[..., 0]
            target = data["target"].to(device)
            u = target[..., 0]
            _, _, metric, _ = metric_func(u_pred, u)
            try:
                metric_val.append(metric.item())
            except:
                metric_val.append(metric)

    return dict(metric=np.mean(metric_val, axis=0))


def run_train(model, loss_func, metric_func,
              train_loader, valid_loader,
              optimizer, lr_scheduler,
              train_batch=None,
              validate_epoch=None,
              epochs=10,
              device="cuda",
              mode='min',
              tqdm_mode='batch',
              patience=10,
              grad_clip=0.999,
              start_epoch: int = 0,
              model_save_path=MODEL_PATH,
              save_mode='state_dict',  # 'state_dict' or 'entire'
              model_name='model.pt',
              result_name='result.pt'):
    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    it = 0

    if patience is None or patience == 0:
        patience = epochs
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    best_val_metric = -np.inf if mode == 'max' else np.inf
    best_val_epoch = None
    save_mode = 'state_dict' if save_mode is None else save_mode
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__)
                             for s in EPOCH_SCHEDULERS)
    tqdm_epoch = False if tqdm_mode == 'batch' else True

    with tqdm(total=end_epoch-start_epoch, disable=not tqdm_epoch) as pbar_ep:
        for epoch in range(start_epoch, end_epoch):
            model.train()
            with tqdm(total=len(train_loader), disable=tqdm_epoch) as pbar_batch:
                for batch in train_loader:
                    if is_epoch_scheduler:
                        loss, _, _ = train_batch(model, loss_func,
                                                 batch, optimizer,
                                                 None, device, grad_clip=grad_clip)
                    else:
                        loss, _, _ = train_batch(model, loss_func,
                                                 batch, optimizer,
                                                 lr_scheduler, device, grad_clip=grad_clip)
                    loss = np.array(loss)
                    loss_epoch.append(loss)
                    it += 1
                    lr = optimizer.param_groups[0]['lr']
                    lr_history.append(lr)
                    desc = f"epoch: [{epoch+1}/{end_epoch}]"
                    if loss.ndim == 0:  # 1 target loss
                        _loss_mean = np.mean(loss_epoch)
                        desc += f" loss: {_loss_mean:.3e}"
                    else:
                        _loss_mean = np.mean(loss_epoch, axis=0)
                        for j in range(len(_loss_mean)):
                            if _loss_mean[j] > 0:
                                desc += f" | loss {j}: {_loss_mean[j]:.3e}"
                    desc += f" | current lr: {lr:.3e}"
                    pbar_batch.set_description(desc)
                    pbar_batch.update()

            loss_train.append(_loss_mean)
            # loss_train.append(loss_epoch)
            loss_epoch = []

            val_result = validate_epoch(
                model, metric_func, valid_loader, device)

            loss_val.append(val_result["metric"])
            val_metric = val_result["metric"].sum()
            if mode == 'max':
                if val_metric > best_val_metric:
                    best_val_epoch = epoch
                    best_val_metric = val_metric
                    stop_counter = 0
                else:
                    stop_counter += 1
            else:
                if val_metric < best_val_metric:
                    best_val_epoch = epoch
                    best_val_metric = val_metric
                    stop_counter = 0
                    if save_mode == 'state_dict':
                        torch.save(model.state_dict(), os.path.join(
                            model_save_path, model_name))
                    else:
                        torch.save(model, os.path.join(
                            model_save_path, model_name))
                    best_model_state_dict = {
                        k: v.to('cpu') for k, v in model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)

                else:
                    stop_counter += 1

            if lr_scheduler and is_epoch_scheduler:
                if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                    lr_scheduler.step(val_metric)
                else:
                    lr_scheduler.step()

            if stop_counter > patience:
                print(f"Early stop at epoch {epoch}")
                break
            if val_result["metric"].ndim == 0:
                desc = color(
                    f"| val metric: {val_metric:.3e} ", color=Colors.blue)
            else:
                metric_0, metric_1 = val_result["metric"][0], val_result["metric"][1]
                desc = color(
                    f"| val metric 1: {metric_0:.3e} ", color=Colors.blue)
                desc += color(f"| val metric 2: {metric_1:.3e} ",
                              color=Colors.blue)
            desc += color(
                f"| best val: {best_val_metric:.3e} at epoch {best_val_epoch+1}", color=Colors.yellow)
            desc += color(f" | early stop: {stop_counter} ", color=Colors.red)
            desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)
            if not tqdm_epoch:
                tqdm.write("\n"+desc+"\n")
            else:
                desc_ep = color("", color=Colors.green)
                if _loss_mean.ndim == 0:  # 1 target loss
                    desc_ep += color(f"| loss: {_loss_mean:.3e} ",
                                     color=Colors.green)
                else:
                    for j in range(len(_loss_mean)):
                        if _loss_mean[j] > 0:
                            desc_ep += color(
                                f"| loss {j}: {_loss_mean[j]:.3e} ", color=Colors.green)
                desc_ep += desc
                pbar_ep.set_description(desc_ep)
                pbar_ep.update()

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                loss_train=np.asarray(loss_train),
                loss_val=np.asarray(loss_val),
                lr_history=np.asarray(lr_history),
                # best_model=best_model_state_dict,
                optimizer_state=optimizer.state_dict()
            )
            save_pickle(result, os.path.join(model_save_path, result_name))
    return result


class ProfileResult:
    def __init__(self, result_file, 
                       num_iters=1,
                       cuda=True) -> None:
        '''
        Hard-coded result computation based on torch.autograd.profiler
        text printout, only works PyTorch 1.8 and 1.9
        '''
        self.columns = ['Name', 'Self CPU %', 'Self CPU',
                        'CPU total %', 'CPU total', 'CPU time avg',
                        'Self CUDA', 'Self CUDA %', 'CUDA total', 'CUDA time avg',
                        'CPU Mem', 'Self CPU Mem', 'CUDA Mem', 'Self CUDA Mem',
                        '# of Calls', 'GFLOPS']
        self.df = pd.read_csv(result_file,
                              delim_whitespace=True,
                              skiprows=range(5),
                              header=None)
        self.num_iters=num_iters
        self.cuda = cuda
        self._clean_df()

    def _clean_df(self):
        df = self.df
        if self.cuda:
            df.loc[:, 16] = df.loc[:, 16].astype(str) + df.loc[:, 17]
            df.loc[:, 14] = df.loc[:, 14].astype(str) + df.loc[:, 15]
        df.loc[:, 12] = df.loc[:, 12].astype(str) + df.loc[:, 13]
        df.loc[:, 10] = df.loc[:, 10].astype(str) + df.loc[:, 11]
        df = df.drop(columns=[11, 13, 15, 17]) if self.cuda else df.drop(columns=[11, 13])
        self.cpu_time_total = df.iloc[-2, 4]
        if self.cuda: self.cuda_time_total = df.iloc[-1, 4]
        df = df[:-3].copy()
        df.columns = self.columns
        self.df = df

    def compute_total_mem(self, col_names):
        total_mems = []
        for col_name in col_names:
            total_mem = 0
            col_vals = self.df[col_name].values
            for val in col_vals:
                if val[-2:] == 'Gb':
                    total_mem += self.get_str_val(val[:-2])
                elif val[-2:] == 'Mb':
                    total_mem += self.get_str_val(val[:-2])/1e3
            total_mems.append(round(total_mem, 2))
        return total_mems

    def compute_total_time(self, col_names):
        total_times = []
        for col_name in col_names:
            total_time = 0
            col_vals = self.df[col_name].values
            for val in col_vals:
                if val[-2:] == 'ms':
                    total_time += float(val[:-2])
                elif val[-2:] == 'us':
                    total_time += float(val[:-2])/1e3
            total_times.append(round(total_time, 2))
        return total_times

    def compute_total(self, col_names):
        totals = []
        for col_name in col_names:
            total = 0
            col_vals = self.df[col_name].values
            for val in col_vals:
                if val[-1].isnumeric():
                    total += float(val)
            totals.append(round(total, 2))
        return totals

    def print_total_mem(self,col_names):
        total_mems = self.compute_total_mem(col_names)
        for i, col_name in enumerate(col_names):
            print(f"{col_name} total: {total_mems[i]} GB")
    
    def print_total(self,col_names):
        totals = self.compute_total(col_names)
        for i, col_name in enumerate(col_names):
            print(f"{col_name} total: {totals[i]}")
    
    def print_total_time(self):
        print(f"# of backprop iters: {self.num_iters}")
        print(f"CPU time total: {self.cpu_time_total}")
        if self.cuda:
            print(f"CUDA time total: {self.cuda_time_total}")

    def print_flop_per_iter(self, flops_col: list):
        totals = self.compute_total(flops_col)
        cuda_time_total = re.findall( r'\d+\.*\d*', self.cuda_time_total)[0]
        for i, col in enumerate(flops_col):
            print(f"{col}*time per iteration: {totals[i]*float(cuda_time_total)/self.num_iters}")

    @staticmethod
    def get_str_val(string):
        if string[0] == '-':
            return  -float(string[1:])
        else:
            return  float(string)


if __name__ == '__main__':
    get_seed(42)
