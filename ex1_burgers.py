from libs import *
SEED = 1127802
DEBUG = False

def main():
    args = get_args_1d()
    
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(args.seed, printout=False)

    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
    train_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=True,
                                   train_portion=0.5,
                                   data_path=data_path,)

    valid_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=False,
                                   valid_portion=100,
                                   data_path=data_path,)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False,
                              drop_last=False, **kwargs)


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

    for arg in vars(args):
        if arg in config.keys():
            config[arg] = getattr(args, arg)

    get_seed(args.seed)
    torch.cuda.empty_cache()
    model = FourierTransformer(**config)
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")

    model_name, result_name = get_model_name(model='burgers',
                                         num_ft_layers=config['num_ft_layers'],
                                         n_hidden=config['n_hidden'],
                                         attention_type=config['attention_type'],
                                         layer_norm=config['layer_norm'],
                                         grid_size=int(2**13//args.subsample),
                                         )
    print(f"Saving model and result in {MODEL_PATH}/{model_name}\n")

    epochs = args.epochs
    lr = args.lr
    h = (1/2**13)*args.subsample
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
                       model_name=model_name,
                       result_name=result_name,
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
