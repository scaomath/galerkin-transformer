from libs_path import *
from libs import *
import argparse
import torch.autograd.profiler as profiler
DEBUG = False


def main():

    # Training settings
    parser = argparse.ArgumentParser(
        description='Memory profiling of various encoder layers')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for profiling (default: 4)')
    parser.add_argument('--attention-type', nargs='+', metavar='attn_type',
                        help='input the attention type for encoders to profile (possile: fourier (alias integral, local), galerkin (alias global), softmax (official PyTorch implementation), linear (standard Q(K^TV) with softmax))',
                        required=True)
    parser.add_argument('--seq-len', type=int, default=1024, metavar='L',
                        help='input sequence length for profiling (default: 1024)')
    parser.add_argument('--dmodel', type=int, default=96, metavar='E',
                        help='input d_model of attention for profiling (default: 96)')
    parser.add_argument('--ndim', type=int, default=1, metavar='dimension',
                        help='input dimension of the Euclidean space (default: 1)')
    parser.add_argument('--num-layers', type=int, default=10, metavar='num_layers',
                        help='input number of encoder layers (default: 10)')
    parser.add_argument('--head', type=int, default=4, metavar='n_head',
                        help='input number of heads in attention for profiling (default: 4)')
    parser.add_argument('--num-iter', type=int, default=1, metavar='k',
                        help='input number of iteration of backpropagations for profiling (default: 1)')
    parser.add_argument('--layer-norm', action='store_true', default=False,
                        help='use the conventional layer normalization')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA in profiling')
    args = parser.parse_args()
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    attn_types = args.attention_type

    for attn_type in attn_types:
        torch.cuda.empty_cache()

        encoder = SimpleTransformerEncoderLayer(d_model=args.dmodel,
                                                 n_head=args.head,
                                                 attention_type=attn_type,
                                                 dim_feedforward=None,
                                                 layer_norm=args.layer_norm,
                                                 attn_norm=not args.layer_norm,
                                                 pos_dim=args.ndim,
                                                 attn_weight=False,)
        encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder) for _ in range(args.num_layers)])
        encoder_layers = encoder_layers.to(device)
        print(
            f"\nModel name: {encoder.__name__}\t Number of params: {get_num_params(encoder_layers)}\n")

        
        pos = torch.randn(args.batch_size, args.seq_len, args.ndim).to(device)
        target = torch.randn(args.batch_size, args.seq_len,
                             args.dmodel).to(device)

        with profiler.profile(profile_memory=True, 
                              use_cuda=cuda, 
                              with_flops=True) as pf:
            with tqdm(total=args.num_iter, disable=(args.num_iter<10)) as pbar:
                for _ in range(args.num_iter):
                    x = torch.randn(args.batch_size, args.seq_len, args.dmodel).to(device)
                    for layer in encoder_layers:
                        x = layer(x, pos)
                    loss = ((x-target)**2).mean()
                    loss.backward()
                    pbar.update()

        sort_by = "self_cuda_memory_usage" if cuda else "self_cpu_memory_usage"
        file_name = os.path.join(SRC_ROOT, f'encoders_pf_{attn_type}.txt')
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
        pf_result.print_flop_per_iter(['GFLOPS'])


if __name__ == '__main__':
    main()
