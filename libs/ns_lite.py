"""
(2+1)D Navier-Stokes equation + Galerkin Transformer
MIT license: Paper2394 authors, NeurIPS 2021 submission.
"""


from .utils import *
from .utils_ft import *
from .ft import *
from .layers import *
from .model import *
import h5py

class NavierStokesDatasetLite(Dataset):
    def __init__(self,
                 data_path=None,
                 train_data=True,
                 train_len=1024,
                 valid_len=200,
                 time_steps_input=10,
                 time_steps_output=10,
                 return_boundary=True,
                 random_state=1127802):
        '''
        PyTorch dataset overhauled for the Navier-Stokes turbulent
        regime data using the vorticity formulation from Li et al 2020
        https://github.com/zongyi-li/fourier_neural_operator

        Baseline:
        FNO2d+time marching network size: 926517
        Galerki transformer+2sc network size: 861617
        original grid size = 64*64, domain = (0,1)^2

        nodes: (N, n, n, T_0:T_1)
        pos: x, y coords flattened, (n*n, 2)
        grid: fine grid, x- and y- coords (n, n, 2)
        targets: solution u_h, (N, n, n, T_1:T_2)
        targets_grad: grad_h u_h, (N, n, n, 2, T_1:T_2)

        '''
        self.data_path = data_path
        self.n_grid = 64  # finest resolution along x-, y- dim
        self.h = 1/self.n_grid
        self.train_data = train_data
        self.time_steps_input = time_steps_input
        self.time_steps_output = time_steps_output
        self.train_len = train_len
        self.valid_len = valid_len
        self.return_boundary = return_boundary
        self.random_state = random_state
        self.eps = 1e-8
        if self.data_path is not None:
            self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        get_seed(self.random_state, printout=False)
        with timer(f"Loading {self.data_path.split('/')[-1]}"):
            data = h5py.File(self.data_path, mode='r')
            x = np.transpose(data['u'])
            a = x[..., :self.time_steps_input]  # (N, n, n, T_0:T_1)
            u = x[...,
                  self.time_steps_input:self.time_steps_input+self.time_steps_output]
            # (N, n, n, T_1:T_2)
            del data, x
            gc.collect()
        if self.train_data:
            a, u = a[:self.train_len], u[:self.train_len]
        else:
            a, u = a[-self.valid_len:], u[-self.valid_len:]
        self.n_samples = len(a)
        self.nodes, self.target, self.target_grad = self.get_data(a, u)

        x = np.linspace(0, 1, self.n_grid)
        y = np.linspace(0, 1, self.n_grid)
        x, y = np.meshgrid(x, y)
        self.grid = np.stack([x, y], axis=-1)
        self.pos = np.c_[x.ravel(), y.ravel()]

    def get_data(self, nodes, targets):
        targets_gradx, targets_grady = self.central_diff(targets, self.h)
        targets_grad = np.stack([targets_gradx, targets_grady], axis=-2)
        # targets = targets[..., None, :]  # (N, n, n, 1, T_1:T_2)
        # nodes = a[..., None, :]
        return nodes, targets, targets_grad

    @staticmethod
    def central_diff(x, h, padding=True):
        # x: (batch, n, n, t)
        if padding:
            x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)),
                       'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (x[:, d:, s:-s] - x[:, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (x[:, s:-s, d:] - x[:, s:-s, :-d])/d  # (N, S_x, S_y, t)

        return grad_x/h, grad_y/h

    def __getitem__(self, idx):
        return dict(node=torch.from_numpy(self.nodes[idx]).float(),
                    pos=torch.from_numpy(self.pos).float(),
                    grid=torch.from_numpy(self.grid).float(),
                    target=torch.from_numpy(self.target[idx]).float(),
                    target_grad=torch.from_numpy(self.target_grad[idx]).float())


class FourierTransformer2DLite(nn.Module):
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


def train_batch_ns(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99):
    optimizer.zero_grad()
    x = data["node"].to(device)
    pos, grid = data['pos'].to(device), data['grid'].to(device)
    u, gradu = data["target"].to(device), data["target_grad"].to(device)

    steps = x.size(-1)
    loss_total = 0
    reg_total = 0
    u_preds = []
    for t in range(steps):
        out_ = model(x, None, pos=pos, grid=grid)
        u_pred = out_['preds']
        u_step = u[..., t:t+1]
        gradu_step = gradu[..., t:t+1]

        # out is (b, n, n, 1)
        loss, reg, _, _ = loss_func(u_pred[..., 0], u_step[..., 0],
                                    targets_prime=gradu_step[..., 0])
        loss = loss + reg
        loss_total += loss
        reg_total += reg.item()

        x = torch.cat((x[..., 1:], u_pred), dim=-1)
        u_preds.append(u_pred)

    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()
    u_preds = torch.cat(u_preds, dim=-1).detach()

    return (loss_total.item()/steps, reg_total/steps), u_preds, None


def validate_epoch_ns(model, metric_func, valid_loader, device):
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            x = data["node"].to(device)
            u = data["target"].to(device)
            pos, grid = data['pos'].to(device), data['grid'].to(device)

            steps = x.size(-1)
            metric_val_step = 0

            for t in range(steps):
                out_ = model(x, None, pos=pos, grid=grid)
                u_pred = out_['preds']
                u_step = u[..., t:t+1]
                
                _, _, metric, _ = metric_func(u_pred[..., 0], u_step[..., 0])
                x = torch.cat((x[..., 1:], u_pred), dim=-1)
                metric_val_step += metric

        metric_val.append(metric_val_step/steps)

    return dict(metric=np.mean(metric_val, axis=0))
