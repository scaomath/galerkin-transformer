import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse import hstack as sparse_hstack
from scipy import interpolate
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import gc
try:
    from utils_ft import *
except:
    from galerkin_transformer.utils_ft import *

import matplotlib.pyplot as plt


class BurgersDataset(Dataset):
    def __init__(self, subsample: int,
                 n_grid_fine=2**13,
                 viscosity: float = 0.1,
                 n_krylov: int = 2,
                 smoother=None,
                 uniform: bool = True,
                 train_data=True,
                 train_portion=0.9,
                 valid_portion=0.1,
                 super_resolution: int = 1,
                 data_path=None,
                 online_features=False,
                 return_edge=False,
                 renormalization=False,
                 return_distance_features=True,
                 return_mass_features=False,
                 return_downsample_grid: bool = True,
                 random_sampling=False,
                 random_state=1127802,
                 debug=False):
        r'''
        PyTorch dataset overhauled for Burger's data from Li et al 2020
        https://github.com/zongyi-li/fourier_neural_operator

        FNO1d network size n_hidden = 64, 16 modes: 549569
        Benchmark: error \approx 1e-2 after 100 epochs after subsampling to 512
        subsampling = 2**3 #subsampling rate
        h = 2**13 // sub #total grid size divided by the subsampling rate

        Periodic BC
        Uniform: 
            - node: f sampled at uniform grid
            - pos: uniform grid, pos encoding 
            - targets: [targets, targets derivative]
        '''
        self.data_path = data_path
        if subsample > 1:
            assert subsample % 2 == 0
        self.subsample = subsample
        self.super_resolution = super_resolution
        self.supsample = subsample//super_resolution
        self.n_grid_fine = n_grid_fine  # finest resolution
        self.n_grid = n_grid_fine // subsample
        self.h = 1/n_grid_fine
        self.uniform = uniform
        self.return_downsample_grid = return_downsample_grid
        self.train_data = train_data
        self.train_portion = train_portion
        self.valid_portion = valid_portion
        self.n_krylov = n_krylov
        self.viscosity = viscosity
        self.smoother = smoother
        self.random_sampling = random_sampling
        self.random_state = random_state
        self.online_features = online_features
        self.edge_features = None  # failsafe
        self.mass_features = None
        self.return_edge = return_edge
        # whether to do Kipf-Weiling renormalization A += I
        self.renormalization = renormalization
        self.return_mass_features = return_mass_features
        self.return_distance_features = return_distance_features
        self.debug = debug
        self._set_seed()
        self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        data = 'train' if self.train_data else 'valid'
        with timer(f"Loading {self.data_path.split('/')[-1]} for {data}."):
            data = loadmat(self.data_path)
            x_data = data['a']
            y_data = data['u']
            del data
            gc.collect()

        train_len, valid_len = self.train_test_split(len(x_data))

        if self.train_data:
            x_data, y_data = x_data[:train_len], y_data[:train_len]
        else:
            x_data, y_data = x_data[-valid_len:], y_data[-valid_len:]

        self.n_samples = len(x_data)

        get_data = self.get_uniform_data if self.uniform else self.get_nonuniform_data
        grid, grid_fine, nodes, targets = get_data(x_data, y_data)

        if not self.online_features and self.return_edge:
            edge_features = []
            mass_features = []
            for i in tqdm(range(self.n_samples)):
                edge, mass = self.get_edge(grid)
                edge_features.append(edge)
                mass_features.append(mass)
            self.edge_features = np.asarray(edge_features, dtype=np.float32)
            self.mass_features = np.asarray(mass_features, dtype=np.float32)

        self.node_features = nodes[..., None] if nodes.ndim == 2 else nodes
        # self.pos = np.c_[grid[...,None], grid[...,None]] #(N, S, 2)
        self.n_features = self.node_features.shape[-1]
        self.pos = grid[..., None] if self.uniform else grid
        self.pos_fine = grid_fine[..., None]
        self.target = targets[..., None] if targets.ndim == 2 else targets

    def _set_seed(self):
        s = self.random_state
        os.environ['PYTHONHASHSEED'] = str(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def get_uniform_data(self, x_data, y_data):
        targets = y_data
        targets_diff = self.central_diff(targets, self.h)

        if self.super_resolution >= 2:
            nodes = x_data[:, ::self.supsample]
            targets = targets[:, ::self.supsample]
            targets_diff = targets_diff[:, ::self.supsample]
        else:
            nodes = x_data[:, ::self.subsample]
            targets = targets[:, ::self.subsample]
            targets_diff = targets_diff[:, ::self.subsample]

        targets = np.stack([targets, targets_diff], axis=2)
        grid = np.linspace(0, 1, self.n_grid)  # subsampled
        # grids = np.asarray([grid for _ in range(self.n_samples)])
        grid_fine = np.linspace(0, 1, self.n_grid_fine//self.supsample)

        return grid, grid_fine, nodes, targets

    @staticmethod
    def central_diff(x, h):
        # x: (batch, seq_len, feats)
        # x: (seq_len, feats)
        # x: (seq_len, )
        # padding is one before feeding to this function
        # TODO: change the pad to u (u padded with the other end b/c periodicity)
        if x.ndim == 1:
            pad_0, pad_1 = x[-2], x[1]  # pad periodic
            x = np.c_[pad_0, x, pad_1]  # left center right
            x_diff = (x[2:] - x[:-2])/2
        elif x.ndim == 2:
            pad_0, pad_1 = x[:, -2], x[:, 1]
            x = np.c_[pad_0, x, pad_1]
            x_diff = (x[:, 2:] - x[:, :-2])/2
            # pad = np.zeros(x_diff.shape[0])

        # return np.c_[pad, x_diff/h, pad]
        return x_diff/h

    @staticmethod
    def laplacian_1d(x, h):
        # standard [1, -2, 1] stencil
        x_lap = (x[1:-1] - x[:-2]) - (x[2:] - x[1:-1])
        # return np.r_[0, x_lap/h**2, 0]
        return x_lap/h**2

    def train_test_split(self, len_data):
        # TODO: change this to random split
        if self.train_portion <= 1:
            train_len = int(self.train_portion*len_data)
        elif 1 < self.train_portion <= len_data:
            train_len = self.train_portion
        else:
            train_len = int(0.8*len_data)

        if self.valid_portion <= 1:
            valid_len = int(self.valid_portion*len_data)
        elif 1 < self.valid_portion <= len_data:
            valid_len = self.valid_portion
        else:
            valid_len = int(0.1*len_data)
        try:
            assert train_len <= len_data - valid_len, \
                f"train len {train_len} be non-overlapping with test len {valid_len}"
        except AssertionError as err:
            print(err)
        return train_len, valid_len

    def get_nonuniform_data(self, x_data, y_data):
        '''
        generate non-uniform data for each sample
        same number of points, but uniformly random chosen
        out:
            - x_data assimilated by u first.
        deprecated
        '''
        targets_u = y_data
        x0, xn = 0, 1
        n_nodes = self.n_grid
        h = self.h
        grid_u = np.linspace(x0, xn, n_nodes)
        grids_u = np.asarray([grid_u for _ in range(self.n_samples)])
        pad_0, pad_n = y_data[:, 0], y_data[:, -1]
        targets_u_diff = self.central_diff(np.c_[pad_0, y_data, pad_n], h)

        grids, nodes, targets, targets_diff = [], [], [], []
        # mesh, function, [solution, solution uniform], derivative
        for i in tqdm(range(self.n_samples)):
            node_fine = x_data[i]  # all 8192 function value f
            _node_fine = np.r_[0, node_fine, 0]  # padding
            node_fine_diff = self.central_diff(_node_fine, h)
            node_fine_lap = self.laplacian_1d(_node_fine, h)
            sampling_density = np.sqrt(
                node_fine_diff**2 + self.viscosity*node_fine_lap**2)
            # sampling_density = np.sqrt(node_fine_lap**2)
            sampling_density = sampling_density[1:-1]
            sampling_density /= sampling_density.sum()

            grid, ix, ix_fine = self.get_grid(sampling_density)

            grids.append(grid)
            node = x_data[i, ix]  # coarse grid
            target, target_diff = y_data[i,
                                         ix_fine], targets_u_diff[i, ix_fine]

            nodes.append(node)
            targets.append(target)
            targets_diff.append(target_diff)

        if self.super_resolution >= 2:
            nodes_u = x_data[:, ::self.supsample]
            targets_u = y_data[:, ::self.supsample]
            targets_u_diff = targets_u_diff[:, ::self.supsample]
        else:
            nodes_u = x_data[:, ::self.subsample]
            targets_u = y_data[:, ::self.subsample]
            targets_u_diff = targets_u_diff[:, ::self.subsample]

        grids = np.asarray(grids)
        nodes = np.asarray(nodes)
        targets = np.asarray(targets)
        targets_diff = np.asarray(targets_diff)
        '''
        target[...,0] = target_u: u on uniform grid
        target[...,1] = targets_u_diff: Du on uniform grid
        target[...,2] = target: u
        target[...,3] = targets_diff: Du

        last dim is for reference
        target[...,4] = nodes_u: f on uniform grid
        '''
        targets = np.stack(
            [targets_u, targets_u_diff, targets, targets_diff, nodes_u], axis=2)

        return grids, grids_u, nodes, targets

    def get_grid(self, sampling_density):
        x0, xn = 0, 1
        sampling_density = None if self.random_sampling else sampling_density
        ix_fine = np.sort(np.random.choice(range(1, self.n_grid_fine-1),
                                           size=self.super_resolution*self.n_grid-2,
                                           replace=False, p=sampling_density))
        ix_fine = np.r_[0, ix_fine, self.n_grid_fine-1]
        ix = ix_fine[::self.super_resolution]
        grid = self.h*ix[1:-1]
        grid = np.r_[x0, grid, xn]
        ix = np.r_[0, ix[1:-1], self.n_grid_fine-1]

        return grid, ix, ix_fine

    def get_edge(self, grid):
        '''
        generate edge features
        '''
        # mass lumping
        weight = np.asarray([self.n_grid for _ in range(
            self.n_grid)]) if self.renormalization else None
        edge = get_laplacian_1d(grid,
                                normalize=True,
                                weight=weight,
                                smoother=self.smoother).toarray().astype(np.float32)
        if self.n_krylov > 1:
            dim = edge.shape  # (S, S)
            dim += (self.n_krylov, )  # (S, S, N_krylov)
            edges = np.zeros(dim)
            edges[..., 0] = edge
            for i in range(1, self.n_krylov):
                edges[..., i] = edge.dot(edges[..., i-1])
        else:
            edges = edge[..., None]

        distance = get_distance_matrix(grid, graph=False)
        mass = get_mass_1d(grid, normalize=False).toarray().astype(np.float32)

        if self.return_mass_features and self.return_distance_features:
            mass = mass[..., None]
            edges = np.concatenate([edges, distance, mass], axis=2)
        elif self.return_distance_features:
            edges = np.concatenate([edges, distance], axis=2)
        return edges, mass

    def __getitem__(self, index):
        '''
        Outputs:
            - pos: coords
            - x: rhs
            - target: solution
        '''
        pos_dim = 1 if self.uniform else 2
        if self.uniform:
            pos = self.pos[:, :pos_dim]
        else:
            pos = self.pos[index, :, :pos_dim]
        grid = pos[..., 0]
        if self.online_features:
            edge = get_laplacian_1d(
                grid, normalize=True).toarray().astype(np.float32)
            if self.n_krylov > 1:
                dim = edge.shape
                dim += (self.n_krylov, )
                edges = np.zeros(dim)
                edges[..., 0] = edge
                for i in range(1, self.n_krylov):
                    edges[..., i] = edge.dot(edges[..., i-1])
            else:
                edges = edge[..., np.newaxis]

            distance = get_distance_matrix(grid, graph=False)
            mass = get_mass_1d(
                grid, normalize=False).toarray().astype(np.float32)
            if self.return_distance_features:
                edges = np.concatenate([edges, distance], axis=2)
            edge_features = torch.from_numpy(edges)
            mass = torch.from_numpy(mass)
        elif not self.online_features and self.return_edge:
            edge_features = torch.from_numpy(self.edge_features[index])
            mass = torch.from_numpy(self.mass_features[index])
        else:
            edge_features = torch.tensor([1.0])
            mass = torch.tensor([1.0])

        if self.return_downsample_grid:
            self.pos_fine = grid[..., None]
        pos_fine = torch.from_numpy(self.pos_fine)
        pos = torch.from_numpy(grid[..., None])
        node_features = torch.from_numpy(self.node_features[index])
        target = torch.from_numpy(self.target[index])
        return dict(node=node_features.float(),
                    pos=pos.float(),
                    grid=pos_fine.float(),
                    edge=edge_features.float(),
                    mass=mass.float(),
                    target=target.float(),)

class UnitGaussianNormalizer:
    def __init__(self, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()
        '''
        modified from utils3.py in 
        https://github.com/zongyi-li/fourier_neural_operator
        Changes:
            - .to() has a return to polymorph the torch behavior
            - naming convention changed to sklearn scalers 
        '''
        self.eps = eps

    def fit_transform(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0)
        return (x - self.mean) / (self.std + self.eps)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return (x * (self.std + self.eps)) + self.mean

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.float().to(device)
            self.std = self.std.float().to(device)
        else:
            self.mean = torch.from_numpy(self.mean).float().to(device)
            self.std = torch.from_numpy(self.std).float().to(device)
        return self

    def cuda(self, device=None):
        assert torch.is_tensor(self.mean)
        self.mean = self.mean.float().cuda(device)
        self.std = self.std.float().cuda(device)
        return self

    def cpu(self):
        assert torch.is_tensor(self.mean)
        self.mean = self.mean.float().cpu()
        self.std = self.std.float().cpu()
        return self


class DarcyDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 inverse_problem=False,
                 normalizer_x=None,
                 normalization=True,
                 renormalization=False,
                 subsample_attn: int = 15,
                 subsample_nodes: int = 1,
                 subsample_inverse: int = 1,
                 subsample_method='nearest',
                 subsample_method_inverse='average',
                 n_krylov: int = 3,
                 uniform: bool = True,
                 train_data=True,
                 train_len=0.9,
                 valid_len=0.0,
                 online_features=False,
                 sparse_edge=False,
                 return_edge=False,
                 return_lap_only=True,
                 return_boundary=True,
                 noise=0,
                 random_state=1127802):
        '''
        PyTorch dataset overhauled for the Darcy flow data from Li et al 2020
        https://github.com/zongyi-li/fourier_neural_operator

        FNO2d network size: 2368001
        original grid size = 421*421
        Laplacian size = (421//subsample) * (421//subsample)
        subsample = 2, 3, 5, 6, 7, 10, 12, 15

        Uniform (update Apr 2021): 
        node: node features, coefficient a, (N, n, n, 1)
        pos: x, y coords, (n_s*n_s, 2)
        grid: fine grid, x- and y- coords (n, n, 2)
        targets: solution u_h, (N, n, n, 1)
        targets_grad: grad_h u_h, (N, n, n, 2)
        edge: Laplacian and krylov, (S, n_sub, n_sub, n_krylov) stored as list of sparse matrices

        '''
        self.data_path = data_path
        self.n_grid_fine = 421  # finest resolution along x-, y- dim
        self.subsample_attn = subsample_attn  # subsampling for attn
        self.subsample_nodes = subsample_nodes  # subsampling for input and output
        self.subsample_inverse = subsample_inverse  # subsample for inverse output
        self.subsample_method = subsample_method  # 'interp' or 'nearest'
        self.subsample_method_inverse = subsample_method_inverse  # 'interp' or 'average'
        # sampling resolution for nodes subsampling along x-, y-
        self.n_grid = int(((self.n_grid_fine - 1)/self.subsample_attn) + 1)
        self.h = 1/self.n_grid_fine
        self.uniform = uniform  # not used
        self.train_data = train_data
        self.train_len = train_len
        self.valid_len = valid_len
        self.n_krylov = n_krylov
        self.return_edge = return_edge
        self.sparse_edge = sparse_edge
        self.normalization = normalization
        self.normalizer_x = normalizer_x  # the normalizer for data
        self.renormalization = renormalization  # whether to add I/h to edge
        self.inverse_problem = inverse_problem  # if True, target is coeff
        self.return_boundary = return_boundary
        # if True, boundary nodes included in grid (-1, n, n) else, (-1, n-2, n-2)
        self.return_lap_only = return_lap_only  # do not return diffusion matrix
        # whether to generate edge features on the go for __getitem__()
        self.online_features = online_features
        self.edge_features = None
        self.mass_features = None
        self.random_state = random_state
        self.eps = 1e-8
        self.noise = noise
        if self.data_path is not None:
            self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        get_seed(self.random_state, printout=False)
        with timer(f"Loading {self.data_path.split('/')[-1]}"):
            try:
                data = loadmat(self.data_path)
                a = data['coeff']  # (N, n, n)
                u = data['sol']  # (N, n, n)
                del data
                gc.collect()
            except FileNotFoundError as e:
                print(r"Please download the dataset from https://github.com/zongyi-li/fourier_neural_operator and put untar them in the data folder.")

        data_len = self.get_data_len(len(a))

        if self.train_data:
            a, u = a[:data_len], u[:data_len]
        else:
            a, u = a[-data_len:], u[-data_len:]
        self.n_samples = len(a)

        nodes, targets, targets_grad = self.get_data(a, u)

        self.coeff = nodes  # un-transformed coeffs
        # pos and elem are already downsampled
        self.pos, self.elem = self.get_grid(self.n_grid)
        self.pos_fine = self.get_grid(self.n_grid_fine,
                                      subsample=self.subsample_nodes,
                                      return_elem=False,
                                      return_boundary=self.return_boundary)

        if self.return_edge and not self.online_features:
            self.edge_features, self.mass_features = self.get_edge(a)
            # (n_s*n_s, 2),  list of a list of csr matrices, list of csr matrix

        # TODO: if processing inverse before normalizer, shapes will be unmatched
        if self.inverse_problem:
            nodes, targets = targets, nodes
            if self.subsample_inverse is not None and self.subsample_inverse > 1:
                n_grid = int(((self.n_grid_fine - 1)/self.subsample_nodes) + 1)
                n_grid_inv = int(
                    ((self.n_grid_fine - 1)/self.subsample_inverse) + 1)
                pos_inv = self.get_grid(n_grid_inv,
                                        return_elem=False,
                                        return_boundary=self.return_boundary)
                if self.subsample_method_inverse == 'average':
                    s_inv = self.subsample_inverse//self.subsample_nodes
                    targets = pooling_2d(targets.squeeze(),
                                         kernel_size=(s_inv, s_inv), padding=True)
                elif self.subsample_method_inverse == 'interp':
                    targets = self.get_interp2d(targets.squeeze(),
                                                n_grid,
                                                n_grid_inv)
                elif self.subsample_method_inverse is None:
                    targets = targets.squeeze()
                self.pos_fine = pos_inv
                targets = targets[..., None]

        if self.train_data and self.normalization:
            self.normalizer_x = UnitGaussianNormalizer()
            self.normalizer_y = UnitGaussianNormalizer()
            nodes = self.normalizer_x.fit_transform(nodes)

            if self.return_boundary:
                _ = self.normalizer_y.fit_transform(x=targets)
            else:
                _ = self.normalizer_y.fit_transform(
                    x=targets[:, 1:-1, 1:-1, :])
        elif self.normalization:
            nodes = self.normalizer_x.transform(nodes)

        if self.noise > 0:
            nodes += self.noise*np.random.randn(*nodes.shape)

        self.node_features = nodes  # (N, n, n, 1)
        self.target = targets  # (N, n, n, 1) of (N, n_s, n_s, 1) if inverse
        self.target_grad = targets_grad  # (N, n, n, 2)

    def get_data_len(self, len_data):
        if self.train_data:
            if self.train_len <= 1:
                train_len = int(self.train_len*len_data)
            elif 1 < self.train_len <= len_data:
                train_len = self.train_len
            else:
                train_len = int(0.8*len_data)
            return train_len
        else:
            if self.valid_len <= 1:
                valid_len = int(self.valid_len*len_data)
            elif 1 < self.valid_len <= len_data:
                valid_len = self.valid_len
            else:
                valid_len = int(0.1*len_data)
            return valid_len

    def get_data(self, a, u):
        # get full resolution data
        # input is (N, 421, 421)
        batch_size = a.shape[0]
        n_grid_fine = self.n_grid_fine
        s = self.subsample_nodes
        n = int(((n_grid_fine - 1)/s) + 1)

        targets = u  # (N, 421, 421)
        if not self.inverse_problem:
            targets_gradx, targets_grady = self.central_diff(
                targets, self.h)  # (N, n_f, n_f)
            targets_gradx = targets_gradx[:, ::s, ::s]
            targets_grady = targets_grady[:, ::s, ::s]

            # targets_gradx = targets_gradx.unsqueeze(-1)
            # targets_grady = targets_grady.unsqueeze(-1)
            # cat = torch.cat if torch.is_tensor(targets_gradx) else np.concatenate
            # grad = cat([targets_gradx, targets_grady], -1) # (N, 2*(n_grid-2)**2)
            targets_grad = np.stack(
                [targets_gradx, targets_grady], axis=-1)  # (N, n, n, 2)
        else:
            targets_grad = np.zeros((batch_size, 1, 1, 2))

        targets = targets[:, ::s, ::s].reshape(batch_size, n, n, 1)

        if s > 1 and self.subsample_method == 'nearest':
            nodes = a[:, ::s, ::s].reshape(batch_size, n, n, 1)
        elif s > 1 and self.subsample_method in ['interp', 'linear', 'average']:
            nodes = pooling_2d(a,
                               kernel_size=(s, s),
                               padding=True).reshape(batch_size, n, n, 1)
        else:
            nodes = a.reshape(batch_size, n, n, 1)

        return nodes, targets, targets_grad

    @staticmethod
    def central_diff(x, h, padding=True):
        # x: (batch, n, n)
        # b = x.shape[0]
        if padding:
            x = np.pad(x, ((0, 0), (1, 1), (1, 1)),
                       'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (x[:, d:, s:-s] - x[:, :-d, s:-s])/d  # (N, S_x, S_y)
        grad_y = (x[:, s:-s, d:] - x[:, s:-s, :-d])/d  # (N, S_x, S_y)

        return grad_x/h, grad_y/h

    @staticmethod
    def get_grid(n_grid, subsample=1, return_elem=True, return_boundary=True):
        x = np.linspace(0, 1, n_grid)
        y = np.linspace(0, 1, n_grid)
        x, y = np.meshgrid(x, y)
        nx = ny = n_grid  # uniform grid
        s = subsample

        if return_elem:
            grid = np.c_[x.ravel(), y.ravel()]  # (n_node, 2)
            elem = []
            for j in range(ny-1):
                for i in range(nx-1):
                    a = i + j*nx
                    b = (i+1) + j*nx
                    d = i + (j+1)*nx
                    c = (i+1) + (j+1)*nx
                    elem += [[a, c, d], [b, c, a]]

            elem = np.asarray(elem, dtype=np.int32)
            return grid, elem
        else:
            if return_boundary:
                x = x[::s, ::s]
                y = y[::s, ::s]
            else:
                x = x[::s, ::s][1:-1, 1:-1]
                y = y[::s, ::s][1:-1, 1:-1]
            grid = np.stack([x, y], axis=-1)
            return grid

    @staticmethod
    def get_grad_tri(grid, elem):
        ve1 = grid[elem[:, 2]]-grid[elem[:, 1]]
        ve2 = grid[elem[:, 0]]-grid[elem[:, 2]]
        ve3 = grid[elem[:, 1]]-grid[elem[:, 0]]
        area = 0.5*(-ve3[:, 0]*ve2[:, 1] + ve3[:, 1]*ve2[:, 0])
        # (# elem, 2-dim vector, 3 vertices)
        Dlambda = np.zeros((len(elem), 2, 3))

        Dlambda[..., 2] = np.c_[-ve3[:, 1]/(2*area), ve3[:, 0]/(2*area)]
        Dlambda[..., 0] = np.c_[-ve1[:, 1]/(2*area), ve1[:, 0]/(2*area)]
        Dlambda[..., 1] = np.c_[-ve2[:, 1]/(2*area), ve2[:, 0]/(2*area)]
        return Dlambda, area

    @staticmethod
    def get_norm_matrix(A, weight=None):
        '''
        A has to be csr
        '''
        if weight is not None:
            A += diags(weight)
        D = diags(A.diagonal()**(-0.5))
        A = (D.dot(A)).dot(D)
        return A

    @staticmethod
    def get_scaler_sizes(n_f, n_c, scale_factor=True):
        factor = np.sqrt(n_c/n_f)
        factor = np.round(factor, 4)
        last_digit = float(str(factor)[-1])
        factor = np.round(factor, 3)
        if last_digit < 5:
            factor += 5e-3
        factor = int(factor/5e-3 + 5e-1 ) * 5e-3
        down_factor = (factor, factor)
        n_m = round(n_f*factor)-1
        up_size = ((n_m, n_m), (n_f, n_f))
        down_size = ((n_m, n_m), (n_c, n_c))
        if scale_factor:
            return down_factor, up_size
        else:
            return down_size, up_size

    @staticmethod
    def get_interp2d(x, n_f, n_c):
        '''
        interpolate (N, n_f, n_f) to (N, n_c, n_c)
        '''
        x_f, y_f = np.linspace(0, 1, n_f), np.linspace(0, 1, n_f)
        x_c, y_c = np.linspace(0, 1, n_c), np.linspace(0, 1, n_c)
        x_interp = []
        for i in range(len(x)):
            xi_interp = interpolate.interp2d(x_f, y_f, x[i])
            x_interp.append(xi_interp(x_c, y_c))
        return np.stack(x_interp, axis=0)

    def get_edge(self, a):
        '''
        Modified from Long Chen's iFEM routine in 2D
        https://github.com/lyc102/ifem
        a: diffusion constant for all, not downsampled
        (x,y), elements downsampled if applicable
        '''
        grid, elem = self.pos, self.elem
        Dphi, area = self.get_grad_tri(grid, elem)
        ks = self.subsample_attn//self.subsample_nodes
        n_samples = len(a)
        online_features = self.online_features
        # (-1, n, n) -> (-1, n_s, n_s)
        a = pooling_2d(a, kernel_size=(ks, ks), padding=True)
        n = len(grid)
        edges = []
        mass = []

        with tqdm(total=n_samples, disable=online_features) as pbar:
            for i in range(n_samples):
                # diffusion coeff
                K = a[i].reshape(-1)  # (n, n) -> (n^2, )
                K_to_elem = K[elem].mean(axis=1)
                # stiffness matrix
                A = csr_matrix((n, n))
                M = csr_matrix((n, n))
                Lap = csr_matrix((n, n))
                for i in range(3):
                    for j in range(3):
                        # $A_{ij}|_{\tau} = \int_{\tau}K\nabla \phi_i\cdot \nabla \phi_j dxdy$
                        Lapij = area*(Dphi[..., i]*Dphi[..., j]).sum(axis=-1)
                        Aij = K_to_elem*Lapij
                        Mij = area*((i == j)+1)/12
                        A += csr_matrix((Aij,
                                        (elem[:, i], elem[:, j])), shape=(n, n))
                        Lap += csr_matrix((Lapij,
                                          (elem[:, i], elem[:, j])), shape=(n, n))
                        M += csr_matrix((Mij,
                                        (elem[:, i], elem[:, j])), shape=(n, n))

                w = np.asarray(
                    M.sum(axis=-1))*self.n_grid**2 if self.renormalization else None
                A, Lap = [self.get_norm_matrix(m, weight=w) for m in (A, Lap)]
                # A, Lap = [self.get_norm_matrix(m) for m in (A, Lap)]
                edge = [A]
                Laps = [Lap]
                if self.n_krylov > 1:
                    for i in range(1, self.n_krylov):
                        edge.append(A.dot(edge[i-1]))
                        Laps.append(Lap.dot(Laps[i-1]))

                edge = Laps if self.return_lap_only else edge + Laps

                edges.append(edge)
                mass.append(M)
                pbar.update()

        return edges, mass

    def __getitem__(self, index):
        '''
        Outputs:
            - pos: x-, y- coords
            - a: diffusion coeff
            - target: solution
            - target_grad, gradient of solution
        '''
        pos_dim = 2
        pos = self.pos[:, :pos_dim]
        if self.return_edge and not self.online_features:
            edges = self.edge_features[index]

            if self.sparse_edge:
                edges = [csr_to_sparse(m) for m in edges]
            else:
                edges = np.asarray(
                    [m.toarray().astype(np.float32) for m in edges])

            edge_features = torch.from_numpy(edges.transpose(1, 2, 0))

            mass = self.mass_features[index].toarray().astype(np.float32)
            mass_features = torch.from_numpy(mass)
        elif self.return_edge and self.online_features:
            a = self.node_features[index].reshape(
                1, self.n_grid_fine, self.n_grid_fine, -1)
            if self.normalization:
                a = self.normalizer_x.inverse_transform(a)
            a = a[..., 0]
            edges, mass = self.get_edge(a)
            edges = np.asarray([m.toarray().astype(np.float32)
                               for m in edges[0]])
            edge_features = torch.from_numpy(edges.transpose(1, 2, 0))

            mass = mass[0].toarray().astype(np.float32)
            mass_features = torch.from_numpy(mass)
        else:
            edge_features = torch.tensor([1.0])
            mass_features = torch.tensor([1.0])
        if self.subsample_attn < 5:
            pos = torch.tensor([1.0])
        else:
            pos = torch.from_numpy(pos)

        grid = torch.from_numpy(self.pos_fine)
        node_features = torch.from_numpy(self.node_features[index])
        coeff = torch.from_numpy(self.coeff[index])
        target = torch.from_numpy(self.target[index])
        target_grad = torch.from_numpy(self.target_grad[index])

        return dict(node=node_features.float(),
                    coeff=coeff.float(),
                    pos=pos.float(),
                    grid=grid.float(),
                    edge=edge_features.float(),
                    mass=mass_features.float(),
                    target=target.float(),
                    target_grad=target_grad.float())


class WeightedL2Loss(_WeightedLoss):
    def __init__(self,
                 dilation=2,  # central diff
                 regularizer=False,
                 h=1/512,  # mesh size
                 beta=1.0,  # L2 u
                 gamma=1e-1,  # \|D(N(u)) - Du\|,
                 alpha=0.0,  # L2 \|N(Du) - Du\|,
                 metric_reduction='L1',
                 periodic=False,
                 return_norm=True,
                 orthogonal_reg=False,
                 orthogonal_mode='global',
                 delta=1e-4,
                 noise=0.0,
                 debug=False
                 ):
        super(WeightedL2Loss, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        assert dilation % 2 == 0
        self.dilation = dilation
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma*h  # H^1
        self.alpha = alpha*h  # H^1
        self.delta = delta*h  # orthongalizer
        self.eps = 1e-8
        # TODO: implement different bc types (Neumann)
        self.periodic = periodic
        self.metric_reduction = metric_reduction
        self.return_norm = return_norm
        self.orthogonal_reg = orthogonal_reg
        self.orthogonal_mode = orthogonal_mode
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    def central_diff(self, x: torch.Tensor, h=None):
        h = self.h if h is None else h
        d = self.dilation  # central diff dilation
        grad = (x[:, d:] - x[:, :-d])/d
        # grad = F.pad(grad, (1,1), 'constant', 0.)  # pad is slow
        return grad/h

    def forward(self, preds, targets,
                preds_prime=None, targets_prime=None,
                preds_latent: list = [], K=None):
        r'''
        all inputs are assumed to have shape (N, L)
        grad has shape (N, L) in 1d, and (N, L, 2) in 2D
        relative error in 
        \beta*\|N(u) - u\|^2 + \alpha*\| N(Du) - Du\|^2 + \gamma*\|D N(u) - Du\|^2
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        on uniform mesh, h can be set to 1
        preds_latent: (N, L, E)
        '''
        batch_size = targets.size(0)

        h = self.h
        if self.noise > 0:
            targets = self._noise(targets, targets.size(-1), self.noise)

        target_norm = h*targets.pow(2).sum(dim=1)

        if targets_prime is not None:
            targets_prime_norm = h*targets_prime.pow(2).sum(dim=1)
        else:
            targets_prime_norm = 1

        loss = self.beta * (h*(preds - targets).pow(2)).sum(dim=1)/target_norm

        if preds_prime is not None and self.alpha > 0:
            grad_diff = h*(preds_prime - K*targets_prime).pow(2)
            loss_prime = self.alpha * grad_diff.sum(dim=1)/targets_prime_norm
            loss += loss_prime

        if self.metric_reduction == 'L2':
            metric = loss.mean().sqrt().item()
        elif self.metric_reduction == 'L1':  # Li et al paper: first norm then average
            metric = loss.sqrt().mean().item()
        elif self.metric_reduction == 'Linf':  # sup norm in a batch
            metric = loss.sqrt().max().item()

        loss = loss.sqrt().mean() if self.return_norm else loss.mean()

        if self.regularizer and self.gamma > 0 and targets_prime is not None:
            preds_diff = self.central_diff(preds)
            s = self.dilation // 2
            regularizer = self.gamma*h*(targets_prime[:, s:-s]
                                        - preds_diff).pow(2).sum(dim=1)/targets_prime_norm

            regularizer = regularizer.sqrt().mean() if self.return_norm else regularizer.mean()

        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)

        if self.orthogonal_reg > 0 and preds_latent:
            ortho = []
            for y_lat in preds_latent:
                if self.orthogonal_mode in ['local', 'fourier']:
                    pred_mm = torch.matmul(
                        y_lat, y_lat.transpose(-2, -1))
                elif self.orthogonal_mode in ['global', 'galerkin', 'linear']:
                    pred_mm = torch.matmul(
                        y_lat.transpose(-2, -1), y_lat)

                with torch.no_grad():
                    mat_dim = pred_mm.size(-1)
                    if self.orthogonal_mode in ['local', 'fourier']:
                        tr = (y_lat**2).sum(dim=-1)
                    elif self.orthogonal_mode in ['global', 'galerkin', 'linear']:
                        tr = (y_lat**2).sum(dim=-2)
                    assert tr.size(-1) == mat_dim
                    diag = [torch.diag(tr[i, :]) for i in range(batch_size)]
                    diag = torch.stack(diag, dim=0)
                ortho.append(
                    self.delta * ((pred_mm - diag)**2).mean(dim=(-1, -2)))
            orthogonalizer = torch.stack(ortho, dim=-1)
            orthogonalizer = orthogonalizer.sqrt().mean(
            ) if self.return_norm else orthogonalizer.mean()
        else:
            orthogonalizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)

        return loss, regularizer, orthogonalizer, metric


class WeightedL2Loss2d(_WeightedLoss):
    def __init__(self,
                 dim=2,
                 dilation=2,  # central diff
                 regularizer=False,
                 h=1/421,  # mesh size
                 beta=1.0,  # L2 u
                 gamma=1e-1,  # \|D(N(u)) - Du\|,
                 alpha=0.0,  # L2 \|N(Du) - Du\|,
                 delta=0.0,  #
                 metric_reduction='L1',
                 return_norm=True,
                 noise=0.0,
                 eps=1e-10,
                 debug=False
                 ):
        super(WeightedL2Loss2d, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        assert dilation % 2 == 0
        self.dilation = dilation
        self.dim = dim
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma  # H^1
        self.alpha = alpha  # H^1
        self.delta = delta*h**dim  # orthogonalizer
        self.eps = eps
        self.metric_reduction = metric_reduction
        self.return_norm = return_norm
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    def central_diff(self, u: torch.Tensor, h=None):
        '''
        u: function defined on a grid (bsz, n, n)
        out: gradient (N, n-2, n-2, 2)
        '''
        bsz = u.size(0)
        h = self.h if h is None else h
        d = self.dilation  # central diff dilation
        s = d // 2  # central diff stride
        if self.dim > 2:
            raise NotImplementedError(
                "Not implemented: dim > 2 not implemented")

        grad_x = (u[:, d:, s:-s] - u[:, :-d, s:-s])/d
        grad_y = (u[:, s:-s, d:] - u[:, s:-s, :-d])/d
        grad = torch.stack([grad_x, grad_y], dim=-1)
        return grad/h

    def forward(self, preds, targets,
                preds_prime=None, targets_prime=None,
                weights=None, K=None):
        r'''
        preds: (N, n, n, 1)
        targets: (N, n, n, 1)
        targets_prime: (N, n, n, 1)
        K: (N, n, n, 1)
        beta * \|N(u) - u\|^2 + \alpha * \| N(Du) - Du\|^2 + \gamma * \|D N(u) - Du\|^2
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        '''
        batch_size = targets.size(0) # for debug only

        h = self.h if weights is None else weights
        d = self.dim
        K = torch.tensor(1) if K is None else K
        if self.noise > 0:
            targets = self._noise(targets, targets.size(-1), self.noise)

        target_norm = targets.pow(2).mean(dim=(1, 2)) + self.eps

        if targets_prime is not None:
            targets_prime_norm = d * \
                (K*targets_prime.pow(2)).mean(dim=(1, 2, 3)) + self.eps
        else:
            targets_prime_norm = 1

        loss = self.beta*((preds - targets).pow(2)
                          ).mean(dim=(1, 2))/target_norm

        if preds_prime is not None and self.alpha > 0:
            grad_diff = (K*(preds_prime - targets_prime)).pow(2)
            loss_prime = self.alpha * \
                grad_diff.mean(dim=(1, 2, 3))/targets_prime_norm
            loss += loss_prime

        if self.metric_reduction == 'L2':
            metric = loss.mean().sqrt().item()
        elif self.metric_reduction == 'L1':  # Li et al paper: first norm then average
            metric = loss.sqrt().mean().item()
        elif self.metric_reduction == 'Linf':  # sup norm in a batch
            metric = loss.sqrt().max().item()

        loss = loss.sqrt().mean() if self.return_norm else loss.mean()

        if self.regularizer and targets_prime is not None:
            preds_diff = self.central_diff(preds)
            s = self.dilation // 2
            targets_prime = targets_prime[:, s:-s, s:-s, :].contiguous()

            if K.ndim > 1:
                K = K[:, s:-s, s:-s].contiguous()

            regularizer = self.gamma * h * ((K * (targets_prime - preds_diff))
                                            .pow(2)).mean(dim=(1, 2, 3))/targets_prime_norm

            regularizer = regularizer.sqrt().mean() if self.return_norm else regularizer.mean()

        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)
        norms = dict(L2=target_norm,
                     H1=targets_prime_norm)

        return loss, regularizer, metric, norms


if __name__ == '__main__':
    subsample = 32
    batch_size = 32
    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')

    train_dataset = BurgersDataset(subsample=subsample,
                                   train_data=True,
                                   return_edge=False,
                                   train_portion=0.1,
                                   data_path=data_path,
                                   random_state=1127802)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=True)
    train_len = len(train_loader)
    print(f"train samples: {len(train_dataset)}")
    sample = next(iter(train_loader))
    for key in sample.keys():
        print(key, "\t", sample[key].shape)
