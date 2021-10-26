from libs_path import *
from libs import *
import argparse
import torch.autograd.profiler as profiler
DEBUG = False


def main():

    # Training settings
    parser = argparse.ArgumentParser(
        description='Memory profiling of various transformers for Example 1')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for profiling (default: 4)')
    parser.add_argument('--attention-type', nargs='+', metavar='attn_type', 
                        help='input the attention type for encoders to profile (possile: fourier (alias integral, local), galerkin (alias global), softmax (official PyTorch implementation), linear (standard Q(K^TV) with softmax))',
                        required=True)
    parser.add_argument('--seq-len', type=int, default=8192, metavar='L',
                        help='input sequence length for profiling (default: 8192)')
    parser.add_argument('--dmodel', type=int, default=96, metavar='E',
                        help='input d_model of attention for profiling (default: 96)')
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
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(SRC_ROOT, r'config.yml')) as f:
        config = yaml.full_load(f)
    config = config['ex1_burgers']
    for arg in vars(args):
        if arg in config.keys():
            config[arg] = getattr(args, arg)
    attn_types = args.attention_type

    for attn_type in attn_types:
        config['attention_type'] = attn_type
        torch.cuda.empty_cache()
        model = SimpleTransformer(**config)
        model = model.to(device)
        print(
            f"\nModel name: {model.__name__}\t Number of params: {get_num_params(model)}")

        node = torch.randn(args.batch_size, args.seq_len, 1).to(device)
        pos = torch.randn(args.batch_size, args.seq_len, 1).to(device)
        target = torch.randn(args.batch_size, args.seq_len, 1).to(device)

        with profiler.profile(profile_memory=True, 
                              use_cuda=cuda, 
                              with_flops=True) as pf:
            with tqdm(total=args.num_iter, disable=(args.num_iter<10)) as pbar:
                for _ in range(args.num_iter):
                    y = model(node, None, pos)
                    y = y['preds']
                    loss = ((y-target)**2).mean()
                    loss.backward()
                    pbar.update()

        sort_by = "self_cuda_memory_usage" if cuda else "self_cpu_memory_usage"
        file_name = os.path.join(SRC_ROOT, f'ex1_{attn_type}.txt')
        with open(file_name, 'w') as f:
            print(pf.key_averages().table(sort_by=sort_by,
                                        row_limit=300,
                                        header=str(model.__name__) +
                                        ' profiling results',
                                        ), file=f)
        pf_result = ProfileResult(file_name, num_iters=args.num_iter, cuda=cuda)
        if cuda:
            pf_result.print_total_mem(['Self CUDA Mem'])
        pf_result.print_total_time()
        pf_result.print_flop_per_iter(['GFLOPS']) # this is in MFLOPs

if __name__ == '__main__':
    main()
