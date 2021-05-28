# Training details for each model


## Example 1: viscous Burgers' equation
On the finest grid, `n=8192`. Note that softmax normalization will result diverging training depending on seed. 

Best Fourier Transformer model:
```bash
python ex1_burgers.py --subsample 1\
                      --attention-type 'fourier'\
                      --xavier-init 0.001 --diag-weight 0.01\
                      --ffn-dropout 0.05\
                      --batch-size 4
```

Best Galerkin Transformer model:
```bash
python ex1_burgers.py --subsample 1\
                      --attention-type 'galerkin'\
                      --xavier-init 0.01 --diag-weight 0.01\
                      --batch-size 4
```


## Memory profiling:
#### Example 1:
```bash
python ex1_memory_profile.py --batch-size 4 --seq-len 8192 --dmodel 96 --num-iter 1 --attention-type 'softmax' 'fourier' 'linear' 'galerkin'
```
Only checking speed not memory
```
python ex1_memory_profile.py --no-memory --batch-size 4 --seq-len 8192 --dmodel 96 --num-iter 1000 --attention-type 'galerkin'
```

#### Example 2:
a 2D model with 128 hidden dimension, using the default `141x141` fine grid, `43x43` coarse grid set up.
```bash
python ex2_memory_profile.py --batch-size 4 --dmodel 128 --attention-type 'softmax' 'fourier' 'linear' 'galerkin'
```
To replicate the results in paper:
```bash
python ex2_memory_profile.py --batch-size 4 --dmodel 128 --attention-type 'softmax' 'fourier' 'linear' 'galerkin' --subsample-nodes 2 --subsample-attn 7 --num-iter 1
```

For real memory usage, use `{$ATTN_TYPE}` can be 'fourier', etc
```
python ex2_memory_profile.py --batch-size 4 --dmodel 128 --attention-type {$ATTN_TYPE} --subsample-nodes 2 --subsample-attn 7 --num-iter 100
```
then open up bash and use `nvidia-smi` to check the active Python process's memory.