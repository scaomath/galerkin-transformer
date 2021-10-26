'''
Use Fourier Features to replace the simple attention in Galerkin Transformer

Code modified from and courtesy of angeloskath@GitHub
https://fast-transformers.github.io/api_docs/fast_transformers/feature_maps/fourier_features.html
'''

from libs_path import *
from libs import *
from math import sqrt, log
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
get_seed(1127802)

subsample = 4
batch_size = 4
data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
train_dataset = BurgersDataset(subsample=subsample,
                               train_data=True,
                               train_portion=0.5,
                               data_path=data_path,)

valid_dataset = BurgersDataset(subsample=subsample,
                               train_data=False,
                               valid_portion=100,
                               data_path=data_path,)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                          pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, drop_last=False,
                          pin_memory=True)

def orthogonal_random_matrix_(w):
    """
    Modified slightly from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/base.py
    to make orthogonal function conforming with PyTorch 1.9 linalg module.

    Initialize the matrix w in-place to compute orthogonal random features.

    The matrix is initialized such that its columns are orthogonal to each
    other (in groups of size `rows`) and their norms is drawn from the
    chi-square distribution with `rows` degrees of freedom (namely the norm of
    a `rows`-dimensional vector distributed as N(0, I)).

    Arguments
    ---------
        w: float tensor of size (rows, columns)
    """
    rows, columns = w.shape
    start = 0
    while start < columns:
        end = min(start+rows, columns)
        block = torch.randn(rows, rows, device=w.device)
        norms = torch.sqrt(torch.einsum("ab,ab->a", block, block))
        Q, _ = torch.linalg.qr(block)
        w[:, start:end] = (
            Q[:, :end-start] * norms[None, :end-start]
        )
        start += rows


class FeatureMap(nn.Module):
    """
    Simplified from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/base.py
    Define the FeatureMap interface.
    """

    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class RandomFourierFeatures(FeatureMap):
    """
    Simplified from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/fourier_features.py
    Removed some customized setup for simplicity.

    Random Fourier Features for the RBF kernel according to [1].

    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """

    def __init__(self, query_dimensions, n_dims=None,
                 orthogonal=False, deterministic_eval=False):
        super(RandomFourierFeatures, self).__init__(query_dimensions)

        self.n_dims = n_dims or query_dimensions
        self.query_dimensions = query_dimensions
        self.orthogonal = orthogonal
        self.softmax_temp = 1/sqrt(query_dimensions)
        self.deterministic_eval = deterministic_eval

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            "omega",
            torch.zeros(self.query_dimensions, self.n_dims//2)
        )

    def new_feature_map(self, device):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        omega = torch.zeros(
            self.query_dimensions,
            self.n_dims//2,
            device=device
        )
        if self.orthogonal:
            orthogonal_random_matrix_(omega)
        else:
            omega.normal_()
        self.register_buffer("omega", omega)

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * sqrt(2/self.n_dims)


class Favor(RandomFourierFeatures):
    """
    Simplified from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/fourier_features.py
    Removed redraw setup for readibility

    Positive orthogonal random features that approximate the softmax kernel.

    Basically implementation of Lemma 1 from "Rethinking Attention with
    Performers".

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the softmax approximation
                     (default: 1/sqrt(query_dimensions))
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        stabilize: bool, If set to True subtract the max norm from the
                   exponentials to make sure that there are no infinities. It
                   is equivalent to a robust implementation of softmax where
                   the max is subtracted before the exponentiation.
                   (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """

    def __init__(self, query_dimensions, n_dims=None,
                 orthogonal=True,
                 deterministic_eval=False):
        super(Favor, self).__init__(query_dimensions,
                                    n_dims=n_dims,
                                    orthogonal=orthogonal,
                                    deterministic_eval=deterministic_eval)

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared/2)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared * 0.5 + 0.5 * log(self.n_dims)

        exp_u1 = torch.exp(u - offset)
        exp_u2 = torch.exp(-u - offset)
        phi = torch.cat([exp_u1, exp_u2], dim=-1)

        return phi


class RandomFourierAttention(nn.Module):
    """
    Modified from 
    https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/attention_layer.py
    positional encoding is now concatinated to the latent vector

    Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, d_model, n_heads, pos_dim=1,
                 eps=1e-6, attention_type='favor',
                 xavier_init=1.0,
                 diagonal_weight=0.0,):
        super(RandomFourierAttention, self).__init__()

        d_k = d_model//n_heads

        if attention_type == 'favor':
            f = Favor.factory(n_dims=d_model)
        elif attention_type == 'rfa':
            f = RandomFourierFeatures.factory(n_dims=d_model)

        self.feature_map = f(d_model)
        self.query_projection = nn.Linear(d_model, d_k * n_heads)
        self.key_projection = nn.Linear(d_model, d_k * n_heads)
        self.value_projection = nn.Linear(d_model, d_k * n_heads)
        self.out_projection = nn.Linear(d_k * n_heads + pos_dim, d_model)
        self.n_heads = n_heads
        self.eps = eps
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        for layer in [self.query_projection, self.key_projection, self.value_projection]:
            self._reset_parameters(layer)

    def _reset_parameters(self, layer):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                if self.diagonal_weight > 0.0:
                    param.data += self.diagonal_weight * \
                        torch.diag(torch.ones(
                            param.size(-1), dtype=torch.float))
            else:
                constant_(param, 0)

    def forward(self, queries, keys, values, pos=None):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: seq_len
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, L, D) The tensor containing the keys
            values: (N, L, D) The tensor containing the values

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward(queries)
        K = self.feature_map.forward(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
        V = V.contiguous().view(N, L, -1)
        return self.out_projection(torch.cat([V, pos], dim=-1))


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='favor',
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 xavier_init: float = 1e-2,
                 diagonal_weight: float = 1e-2,
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False,
                 ):
        super(TransformerEncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        self.attn = RandomFourierAttention(d_model, n_head,
                                           attention_type=attention_type,
                                           xavier_init=xavier_init,
                                           diagonal_weight=diagonal_weight,)

        self.d_model = d_model
        self.n_head = n_head
        self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.debug = debug

    def forward(self, x, pos=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional 
            information if coords are in features
        '''
        att_output = self.attn(x, x, x, pos=pos)

        x = x + self.dropout1(att_output)
        x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        x = self.layer_norm2(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleTransformer, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()

    def forward(self, node, edge, pos, grid=None):
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
        node = torch.cat([node, pos], dim=-1)
        x = self.feat_extract(node, edge)

        for encoder in self.encoder_layers:
            x = encoder(x, pos)

        x = self.dpo(x)
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
        encoder_layer = TransformerEncoderLayer(d_model=self.n_hidden,
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


config = defaultdict(lambda: None,
                     node_feats=1+1,
                     pos_dim=1,
                     n_targets=1,
                     n_hidden=96,
                     num_feat_layers=0,
                     num_encoder_layers=4,
                     n_head=1,
                     dim_feedforward=192,
                     attention_type='favor', # 'favor' or 'rfa'
                     feat_extract_type=None,
                     xavier_init=0.01,
                     diagonal_weight=0.0,
                     layer_norm=True,
                     attn_norm=False,
                     return_attn_weight=False,
                     return_latent=False,
                     decoder_type='ifft',
                     freq_dim=48,  # hidden dim in the frequency domain
                     num_regressor_layers=2,  # number of spectral layers
                     fourier_modes=16,  # number of fourier modes
                     spacial_dim=1,
                     spacial_fc=False,
                     dropout=0.0,
                     encoder_dropout=0.1,
                     decoder_dropout=0.0,
                     debug=False,
                     )

torch.cuda.empty_cache()
model = SimpleTransformer(**config)
model = model.to(device)

print(f"\nNumber of params: {get_num_params(model)}")

epochs = 100
lr = 1e-3
h = (1/2**13)*subsample
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, final_div_factor=1e4,
                       steps_per_epoch=len(train_loader), epochs=epochs)

loss_func = WeightedL2Loss(regularizer=True, h=h, gamma=0.1)

metric_func = WeightedL2Loss(regularizer=False, h=h)

result = run_train(model, loss_func, metric_func,
                   train_loader, valid_loader,
                   optimizer, scheduler,
                   train_batch=train_batch_burgers,
                   validate_epoch=validate_epoch_burgers,
                   epochs=epochs,
                   patience=None,
                   tqdm_mode='epoch',
                   mode='min',
                   device=device)

'''
Performer:
- Without concat coordinate: 5.593e-03
    

Concat positional encoding in every head:
- without new diagonal-dominant initialization: 1.676e-03
- Using our new initialization: 1.582e-03

Random Fourier Feature (basically a simple non-orthogonal ver of Performer)
- without new diagonal-dominant initialization: 1.715e-02
'''
