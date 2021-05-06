from libs import *

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

SEED = 1127802
get_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = False

subsample = 4
batch_size = 8

def get_data():

    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
    train_dataset = BurgersDataset(subsample=subsample,
                                   train_data=True,
                                   train_portion=0.5,
                                   data_path=data_path,
                                   random_state=SEED)

    valid_dataset = BurgersDataset(subsample=subsample,
                                   train_data=False,
                                   valid_portion=100,
                                   data_path=data_path,
                                   random_state=SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, drop_last=False,
                              pin_memory=True)
    return train_loader, valid_loader


def main():
    train_loader, valid_loader = get_data()
    sample = next(iter(train_loader))
    for key in sample.keys():
        print(key, "\t", sample[key].shape)

    f = sample['node']
    pos = sample['pos']
    u = sample['target']

    if is_interactive():
        _, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
        axes = axes.reshape(-1)
        indexes = np.random.choice(range(4), size=4, replace=False)
        for i, ix in enumerate(indexes):
            axes[i].plot(pos[ix], f[ix], label='input')
            axes[i].plot(pos[ix], u[ix, :, 0], label='target')
            axes[i].plot(pos[ix, 1:-1], u[ix, 1:-1, 1],
                         label='target derivative')
            axes[i].legend()

    with open(r'./config.yml') as f:
        config = yaml.full_load(f)
    test_name = os.path.basename(__file__).split('.')[0]
    config = config[test_name]

    torch.cuda.empty_cache()
    model = FourierTransformer(**config)
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
