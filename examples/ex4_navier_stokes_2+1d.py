"""
(2+1)D Navier-Stokes equation + Galerkin Transformer
MIT license: Paper2394 authors, NeurIPS 2021 submission.
"""
from libs_path import *
from libs import *
from libs.ns_lite import *
get_seed(1127802)

data_path = os.path.join(DATA_PATH, 'ns_V1000_N5000_T50.mat')
train_dataset = NavierStokesDatasetLite(data_path=data_path,
                                        train_data=True,)
valid_dataset = NavierStokesDatasetLite(data_path=data_path,
                                        train_data=False,)
batch_size = 4
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

config = defaultdict(lambda: None,
                     node_feats=10+2,
                     pos_dim=2,
                     n_targets=1,
                     n_hidden=48,  # attention's d_model
                     num_feat_layers=0,
                     num_encoder_layers=4,
                     n_head=1,
                     dim_feedforward=96,
                     attention_type='galerkin',
                     feat_extract_type=None,
                     xavier_init=0.01,
                     diagonal_weight=0.01,
                     layer_norm=True,
                     attn_norm=False,
                     return_attn_weight=False,
                     return_latent=False,
                     decoder_type='ifft',
                     freq_dim=20,  # hidden dim in the frequency domain
                     num_regressor_layers=2,  # number of spectral layers
                     fourier_modes=12,  # number of Fourier modes
                     spacial_dim=2,
                     spacial_fc=False,
                     dropout=0.0,
                     encoder_dropout=0.0,
                     decoder_dropout=0.0,
                     ffn_dropout=0.05,
                     debug=False,
                     )

torch.cuda.empty_cache()
model = FourierTransformer2DLite(**config)
print(get_num_params(model))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

epochs = 100
lr = 1e-3
h = 1/64
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, final_div_factor=1e4,
                       steps_per_epoch=len(train_loader), epochs=epochs)

loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=0.1)

metric_func = WeightedL2Loss2d(regularizer=False, h=h)

result = run_train(model, loss_func, metric_func,
                   train_loader, valid_loader,
                   optimizer, scheduler,
                   train_batch=train_batch_ns,
                   validate_epoch=validate_epoch_ns,
                   epochs=epochs,
                   patience=None,
                   tqdm_mode='batch',
                   mode='min',
                   device=device)
"""
4 GT layers: 48 d_model
2 SC layers: 20 d_model for spectral conv with 12 Fourier modes
Total params: 862049

diag 0 + xavier 1e-2, encoder dp = ffn dp = 5e-2
    3.406e-03 at epoch 99

diag 1e-2 + xavier 1e-2, encoder dp 0, ffn dp = 5e-2
    3.078e-03 at epoch 100
"""
