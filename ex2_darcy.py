from libs import *
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

SEED = 1127802
DEBUG = False

subsample_attn = 10
subsample_nodes = 3
batch_size = 8
val_batch_size = 4


def train_batch_darcy(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99):
    optimizer.zero_grad()
    a, x, edge = data["coeff"].to(device), data["node"].to(
        device), data["edge"].to(device)
    pos, grid = data['pos'].to(device), data['grid'].to(device)
    u, gradu = data["target"].to(device), data["target_grad"].to(device)

    out_ = model(x, edge, pos=pos, grid=grid)
    if isinstance(out_, dict):
        out = out_['preds']
    elif isinstance(out_, tuple):
        out = out_[-1]

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
                out = out_[-1]
            u_pred = out[..., 0]
            target = data["target"].to(device)
            u = target[..., 0]
            _, _, metric, _ = metric_func(u_pred, u)
            try:
                metric_val.append(metric.item())
            except:
                metric_val.append(metric)

    return dict(metric=np.mean(metric_val, axis=0))

def get_data(train_len=1024,
             valid_len=100,
             random_state=SEED,
             batch_size=batch_size,
             val_batch_size=val_batch_size,
             subsample_attn=subsample_attn,
             subsample_nodes=subsample_nodes,
             **kwargs):

    train_path = os.path.join(DATA_PATH, 'piececonst_r421_N1024_smooth1.mat')
    test_path = os.path.join(DATA_PATH, 'piececonst_r421_N1024_smooth2.mat')
    train_dataset = DarcyDataset(data_path=train_path,
                                 subsample_attn=subsample_attn,
                                 subsample_nodes=subsample_nodes,
                                 train_data=True,
                                 online_features=True if DEBUG else False,
                                 train_len=train_len if not DEBUG else 0.05,
                                 random_state=random_state)

    valid_dataset = DarcyDataset(data_path=test_path,
                                 normalizer_x=train_dataset.normalizer_x,
                                 subsample_attn=subsample_attn,
                                 subsample_nodes=subsample_nodes,
                                 train_data=False,
                                 online_features=True if DEBUG else False,
                                 valid_len=valid_len,
                                 random_state=random_state)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False,
                              drop_last=False, **kwargs)
    return train_loader, valid_loader, train_dataset


def main():

    # Training settings
    parser = argparse.ArgumentParser(
        description='Example 2: Darcy interface flow')
    parser.add_argument('--subsample-nodes', type=int, default=3, metavar='subsample',
                        help='input fine grid sampling from 421x421 (default: 3 i.e., 141x141 grid)')
    parser.add_argument('--subsample-attn', type=int, default=10, metavar='subsample_attn',
                        help='input coarse grid sampling from 421x421 (default: 10 i.e., 43x43 grid)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--val-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for validation (default: 4)')
    parser.add_argument('--attn-type', type=str, default='galerkin', metavar='attn_type',
                        help='input attention type for encoders (possile: fourier (alias integral, local), galerkin (alias global), softmax (official PyTorch implementation), linear (standard Q(K^TV) with softmax), default: galerkin)')
    parser.add_argument('--xavier-init', type=float, default=1e-2, metavar='xavier_init',
                        help='input Xavier initialization strength for Q,K,V weights (default: 0.01)')
    parser.add_argument('--diag-weight', type=float, default=1e-2, metavar='diag_weight',
                        help='input diagonal weight initialization strength for Q,K,V weights (default: 0.01)')
    parser.add_argument('--reg-layernorm', action='store_true', default=False,
                        help='use the conventional layer normalization')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='max learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=5e-1, metavar='regularizer',
                        help='strength of gradient regularizer (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=SEED, metavar='Seed',
                        help='random seed (default: 1127802)')
    args = parser.parse_args()
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}

    torch.manual_seed(seed=args.seed)
    torch.cuda.manual_seed(seed=args.seed)

    train_loader, valid_loader, train_dataset = get_data(
        subsample_nodes=args.subsample_nodes,
        subsample_attn=args.subsample_attn,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        **kwargs)
    n_grid = int(((421 - 1)/args.subsample_nodes) + 1)
    n_grid_c = int(((421 - 1)/args.subsample_attn) + 1)
    downsample, upsample = DarcyDataset.get_scaler_sizes(n_grid, n_grid_c)

    sample = next(iter(train_loader))

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)
    print('='*(40 + len('Data loader batch')+2))

    if is_interactive():
        idx = 3
        a = sample['node']
        u = sample['target']
        elem = train_dataset.elem
        node = train_dataset.pos
        ah = F.interpolate(a[..., 0].unsqueeze(1), size=(
            n_grid_c, n_grid_c), mode='bilinear', align_corners=True)
        uh = F.interpolate(u[..., 0].unsqueeze(1), size=(
            n_grid_c, n_grid_c), mode='bilinear', align_corners=True)
        ah = ah[idx].numpy().reshape(-1)
        uh = uh[idx].numpy().reshape(-1)
        showsolution(node, elem, uh, width=600, height=500)

        fig, ax = plt.subplots(figsize=(10, 10))
        ah_plot = ax.imshow(ah.reshape(n_grid_c, n_grid_c), cmap='RdBu')
        fig.colorbar(ah_plot, ax=ax, anchor=(0, 0.3), shrink=0.8)

    with open(r'./config.yml') as f:
        config = yaml.full_load(f)
    test_name = os.path.basename(__file__).split('.')[0]
    config = config[test_name]
    config['attention_type'] = args.attn_type
    config['diagonal_weight'] = args.diag_weight
    config['xavier_init'] = args.xavier_init
    config['normalizer'] = train_dataset.normalizer_y.to(device)
    config['downscaler_size'] = downsample
    config['upscaler_size'] = upsample
    config['layer_norm'] = args.reg_layernorm
    config['attn_norm'] = not args.reg_layernorm

    torch.cuda.empty_cache()
    model = FourierTransformer2D(**config)
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}\n")

    epochs = args.epochs
    lr = args.lr
    h = (1/421)*args.subsample_nodes
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, final_div_factor=1e4,
                           steps_per_epoch=len(train_loader), epochs=epochs)

    loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=args.gamma)

    metric_func = WeightedL2Loss2d(regularizer=False, h=h)

    result = run_train(model, loss_func, metric_func,
                       train_loader, valid_loader,
                       optimizer, scheduler,
                       train_batch=train_batch_darcy,
                       validate_epoch=validate_epoch_darcy,
                       epochs=epochs,
                       patience=None,
                       tqdm_mode='epoch',
                       device=device)

    plt.figure(1)
    loss_train = result['loss_train']
    loss_val = result['loss_val']
    plt.semilogy(loss_train[:, 0], label='train')
    plt.semilogy(loss_val, label='valid')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
