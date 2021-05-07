from libs import *
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

SEED = 1127802
DEBUG = False

subsample = 4
batch_size = 8
val_batch_size = 4

def get_data(train_portion=1024, 
             valid_portion=100, 
             random_state=SEED,
             batch_size=batch_size,
             val_batch_size=val_batch_size,
             subsample=subsample,
             **kwargs):

    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
    train_dataset = BurgersDataset(subsample=subsample,
                                   train_data=True,
                                   train_portion=train_portion,
                                   data_path=data_path,
                                   random_state=random_state)

    valid_dataset = BurgersDataset(subsample=subsample,
                                   train_data=False,
                                   valid_portion=valid_portion,
                                   data_path=data_path,
                                   random_state=random_state)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, 
                              drop_last=False,**kwargs)
    return train_loader, valid_loader


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Example 1: Burgers equation')
    parser.add_argument('--subsample', type=int, default=4, metavar='subsample',
                        help='input sampling from 8192 (default: 4 i.e., 2048 grid)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='n_b',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--val-batch-size', type=int, default=4, metavar='n_b',
                        help='input batch size for validation (default: 4)')
    parser.add_argument('--attn-type', type=str, default='fourier', metavar='attn_type',
                        help='input attention type for encoders (possile: fourier (alias integral, local), galerkin (alias global), softmax (official PyTorch implementation), linear (standard Q(K^TV) with softmax), default: fourier)')
    parser.add_argument('--xavier-init', type=float, default=1e-2, metavar='xavier_init',
                        help='input Xavier initialization strength for Q,K,V weights (default: 0.01)')
    parser.add_argument('--diag-weight', type=float, default=1e-2, metavar='diag_weight',
                        help='input diagonal weight initialization strength for Q,K,V weights (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='max learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=1e-1, metavar='regularizer',
                        help='strength of gradient regularizer (default: 0.1)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables GPU training')
    parser.add_argument('--seed', type=int, default=SEED, metavar='Seed',
                        help='random seed (default: 1127802)')
    args = parser.parse_args()
    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}

    get_seed(args.seed)

    train_loader, valid_loader = get_data(subsample=args.subsample,
                                          batch_size=args.batch_size,
                                          val_batch_size=args.val_batch_size,
                                          **kwargs)
    sample = next(iter(train_loader))

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)
    print('='*(40 + len('Data loader batch')+2))

    if is_interactive():
        u0 = sample['node']
        pos = sample['pos']
        u = sample['target']
        _, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
        axes = axes.reshape(-1)
        indexes = np.random.choice(range(4), size=4, replace=False)
        for i, ix in enumerate(indexes):
            axes[i].plot(pos[ix], u0[ix], label='input')
            axes[i].plot(pos[ix], u[ix, :, 0], label='target')
            axes[i].plot(pos[ix, 1:-1], u[ix, 1:-1, 1],
                         label='target derivative')
            axes[i].legend()

    with open(r'./config.yml') as f:
        config = yaml.full_load(f)
    test_name = os.path.basename(__file__).split('.')[0]
    config = config[test_name]
    config['attention_type'] = args.attn_type
    config['diagonal_weight'] = args.diag_weight
    config['xavier_init'] = args.xavier_init

    torch.cuda.empty_cache()
    model = FourierTransformer(**config)
    model = model.to(device)
    print(f"\nNumber of params: {get_num_params(model)}")

    epochs = args.epochs
    lr = args.lr
    h = (1/2**13)*subsample
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, final_div_factor=1e4,
                           steps_per_epoch=len(train_loader), epochs=epochs)

    loss_func = WeightedL2Loss(regularizer=True, h=h, gamma=args.gamma)

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
