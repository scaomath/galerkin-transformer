from libs import *
import argparse
import torch.autograd.profiler as profiler
DEBUG = False


def main():

    # Training settings
    parser = argparse.ArgumentParser(
        description='Memory profiling for Fourier and Galerkin transformers')
    parser.add_argument('--batch-size', type=int, default=4, metavar='n_b',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--num-iter', type=int, default=1, metavar='k',
                        help='input number of iteration for profiling (default: 1)')
    parser.add_argument('--reg-layernorm', action='store_true', default=False,
                        help='use the conventional layer normalization')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, r'config.yml')) as f:
        config = yaml.full_load(f)
    config = config['ex1_burgers']
    config['layer_norm'] = args.reg_layernorm
    config['attn_norm'] = not args.reg_layernorm

    for attn_type in ['softmax', 'fourier', 'linear', 'galerkin',]:
        config['attention_type'] = attn_type
        torch.cuda.empty_cache()
        model = FourierTransformer(**config)
        model = model.to(device)
        print(
            f"\nModel name: {model.__name__}\t Number of params: {get_num_params(model)}\n")

        node = torch.randn(args.batch_size, 8192, 1).to(device)
        pos = torch.randn(args.batch_size, 8192, 1).to(device)
        target = torch.randn(args.batch_size, 8192, 1).to(device)

        with profiler.profile(profile_memory=True, use_cuda=True,) as pf:
            for _ in range(args.num_iter):
                y = model(node, None, pos)
                y = y['preds']
                loss = ((y-target)**2).mean()
                loss.backward()

        print(pf.key_averages().table(sort_by="self_cuda_memory_usage",
                                        row_limit=300,
                                        header=str(model.__name__) + ' profiling results',
                                        ))


if __name__ == '__main__':
    main()
