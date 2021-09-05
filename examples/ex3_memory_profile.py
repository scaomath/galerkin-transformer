from libs_path import *
from libs import *
import argparse
import torch.autograd.profiler as profiler
DEBUG = False


def main():
    parser = argparse.ArgumentParser(
        description='Memory profiling of various transformers for Example 2')
    parser.add_argument('--attention-type', nargs='+', metavar='attn_type', 
                        help='input the attention type for encoders to profile (possile: fourier (alias integral, local), galerkin (alias global), softmax (official PyTorch implementation), linear (standard Q(K^TV) with softmax))',
                        required=True)
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for profiling (default: 4)')
    parser.add_argument('--subsample-nodes', type=int, default=2, metavar='subsample',
                        help='input fine grid sampling from 421x421 (default: 2 i.e., 211x211 grid)')
    parser.add_argument('--subsample-attn', type=int, default=6, metavar='subsample_attn',
                        help='input coarse grid sampling from 421x421 (default: 6 i.e., 71x71 grid)')
    parser.add_argument('--dmodel', type=int, default=192, metavar='E',
                        help='input d_model of attention for profiling (default: 64)')
    parser.add_argument('--num-iter', type=int, default=1, metavar='k',
                        help='input number of iteration of backpropagations for profiling (default: 1)')
    parser.add_argument('--layer-norm', action='store_true', default=False,
                        help='use the conventional layer normalization')
    parser.add_argument('--no-memory', action='store_true', default=False,
                        help='disables memory profiling')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA in profiling')
    args = parser.parse_args()
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)

    n_grid = int(((421 - 1)/args.subsample_nodes) + 1)
    n_grid_c = int(((421 - 1)/args.subsample_attn) + 1)
    downsample, _ = DarcyDataset.get_scaler_sizes(n_grid, n_grid_c)

    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, r'config.yml')) as f:
        config = yaml.full_load(f)
    config = config['ex3_darcy_inv']
    for arg in vars(args):
        if arg in config.keys():
            config[arg] = getattr(args, arg)
    config['downscaler_size'] = downsample
    config['upscaler_size'] = ((n_grid_c, n_grid_c), (n_grid_c, n_grid_c))
    attn_types = args.attention_type

    for attn_type in attn_types:
        config['attention_type'] = attn_type
        torch.cuda.empty_cache()
        model = FourierTransformer2D(**config)
        model = model.to(device)
        print(
            f"\nModel name: {model.__name__}\t Number of params: {get_num_params(model)}")

        node = torch.randn(args.batch_size, n_grid, n_grid, 1).to(device)
        pos = torch.randn(args.batch_size, n_grid_c**2, 2).to(device)
        target = torch.randn(args.batch_size, n_grid_c, n_grid_c, 1).to(device)
        grid = torch.randn(args.batch_size, n_grid_c, n_grid_c, 2).to(device)

        with profiler.profile(profile_memory=True, use_cuda=cuda,) as pf:
            with tqdm(total=args.num_iter, disable=(args.num_iter<10)) as pbar:
                for _ in range(args.num_iter):
                    y = model(node, None, pos, grid)
                    y = y['preds']
                    loss = ((y-target)**2).mean()
                    loss.backward()
                    pbar.update()

        sort_by = "self_cuda_memory_usage" if cuda else "self_cpu_memory_usage"
        file_name = os.path.join(SRC_ROOT, f'ex2_{attn_type}.txt')
        with open(file_name, 'w') as f:
            print(pf.key_averages().table(sort_by=sort_by,
                                        row_limit=300,
                                        header=str(model.__name__) +
                                        ' profiling results',
                                        ), file=f)
        pf_result = ProfileResult(file_name, num_iters=args.num_iter, cuda=cuda)
        pf_result.print_total_mem(['Self CUDA Mem'])
        pf_result.print_total_time()


if __name__ == '__main__':
    main()
