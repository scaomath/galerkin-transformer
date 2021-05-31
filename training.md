# Training details for each model
- `--no-cuda`: use CPU, not recommended.
- `--reg-layernorm`: use the conventional layer normalization scheme that kills all scalings.
- `--batch-size` + a number.
- `--attention-type`: `'softmax'`,  `'fourier'`,  `'linear'`, or  `'galerkin'`.
- `--xavier-init`: gain for Xavier init for `W^{Q,K,V}`.
- `--diag-weight`: a small diagonal matrix is added to the initialization of `W^{Q,K,V}`, recommended value is `1e-2`.
- `--encoder-dropout`: dropout for the attention weights.
- `--ffn-dropout`: dropout for the FFN in attention blocks.
- `--decoder-dropout`: dropout in the decoder block.
- `--gamma`: the strength of the $H^1$-seminorm regularizer, `0.1` in Example 1, and `0.5` in Example 2, when the target is not a smooth function, set this to 0.
- `--seed`: RNG, default `1127802`.


## Remarks on various comparisons
- If we want to compare with the regular layer normalization scheme, just add `--reg-layernorm` in the end, the new scale-preserving layer normalization will be automatically disabled.
- If we want to compare the softmax normalized counterparts, just change `'galerkin'` to `'linear'`, `'fourier'` to `'softmax'`, and the setting should be carried over. 
- If we want to use the default setting of the original Transformer, please use `--xavier-init 1 --diag-weight 0 --ffn-dropout 0.1 --encoder-dropout 0.1` in the arguments.
- By default, the noise for the inverse coefficient identification problem is 0.01. If we want to have a specific noise, please `--noise $NOISE`.
- If we want to compare with the FNO baselines, please clone the repo at https://github.com/zongyi-li/fourier_neural_operator to local, and change the scheduler in `fourier_1d.py` and `fourier_2d.py` to:
    ```python
    epochs = 100
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, div_factor=1e4, final_div_factor=1e4,
                            steps_per_epoch=len(train_loader), epochs=epochs)
    ```



# Example 1: viscous Burgers' equation
On the finest grid, `n=8192`. It is recommended using the diagonal dominant initialization for even the classic softmax normalized Transformer, as the regular Xavier initialization with gain `1` will result diverging training depending on seed.

Fourier Transformer model:
```bash
python ex1_burgers.py --subsample 1 --attention-type 'fourier' --xavier-init 0.001 --diag-weight 0.01  --ffn-dropout 0.05 --batch-size 4
```

Galerkin Transformer model:
```bash
python ex1_burgers.py --subsample 1 --attention-type 'galerkin' --xavier-init 0.01 --diag-weight 0.01 --batch-size 4
```

Subsample 4, i.e., `n=2048`.

```bash
python ex1_burgers.py --subsample 4 --attention-type 'fourier' --xavier-init 0.001 --diag-weight 0.01  --ffn-dropout 0.05 --batch-size 4
```

```bash
python ex1_burgers.py --subsample 4 --attention-type 'galerkin' --xavier-init 0.01 --diag-weight 0.01 --batch-size 4
```



# Example 2:
`141x141` fine grid, `43x43` coarse grid: 

```bash
python ex2_darcy.py --subsample-attn 10 --subsample-nodes 3 --attention-type 'galerkin' --xavier-init 0.01 --diag-weight 0.01
```

```bash
python ex2_darcy.py --subsample-attn 10 --subsample-nodes 3 --attention-type 'fourier' --xavier-init 0.01 --diag-weight 0.01 --ffn-dropout 0.1 --encoder-dropout 0.1 --lr 0.0005
```

`211x211` fine grid, `61x61` coarse grid:
```bash
python ex2_darcy.py --subsample-attn 7 --subsample-nodes 2 --attention-type 'galerkin' --xavier-init 0.01 --diag-weight 0.01 --ffn-dropout 0.05 --encoder-dropout 0.1
```

Using Fourier attention is not recommended (slow due to the `n^2`-complexity of local attention):
```bash
python ex2_darcy.py --subsample-attn 7 --subsample-nodes 2 --attention-type 'fourier' --xavier-init 0.001 --diag-weight 0.01 --ffn-dropout 0.1 --encoder-dropout 0.05 --lr 0.0005
```

# Example 3:
On a `211x211` fine grid, `71x71` coarse grid:
```bash
python ex3_darcy_inv.py --attention-type 'galerkin' --xavier-init 0.01 --diag-weight 0.01
```

```bash
python ex3_darcy_inv.py --attention-type 'fourier' --xavier-init 0.01 --diag-weight 0.01 --ffn-dropout 0.1 --lr 0.0005
```

`141x141` fine grid, `36x36` coarse grid:
```bash
python ex3_darcy_inv.py --subsample-attn 12 --subsample-nodes 3 --attention-type 'galerkin' --xavier-init 0.01 --diag-weight 0.01
```

```bash
python ex3_darcy_inv.py --subsample-attn 12 --subsample-nodes 3 --attention-type 'fourier' --xavier-init 0.01 --diag-weight 0.01 --ffn-dropout 0.1 --lr 0.0005
```


# Memory profiling:

## Example 1:
```bash
python ex1_memory_profile.py --batch-size 4 --seq-len 8192 --dmodel 96 --num-iter 1 --attention-type 'softmax' 'fourier' 'linear' 'galerkin'
```
Only checking speed not memory
```
python ex1_memory_profile.py --no-memory --batch-size 4 --seq-len 8192 --dmodel 96 --num-iter 1000 --attention-type 'galerkin'
```

## Example 2:
a 2D model with 128 hidden dimension, using the default `141x141` fine grid, `43x43` coarse grid set up.
```bash
python ex2_memory_profile.py --batch-size 4 --dmodel 128 --attention-type 'softmax' 'fourier' 'linear' 'galerkin'
```
To replicate the results in paper:
```bash
python ex2_memory_profile.py --batch-size 4 --dmodel 128 --attention-type 'softmax' 'fourier' 'linear' 'galerkin' --subsample-nodes 2 --subsample-attn 7 --num-iter 1
```

For real memory usage, use `{$ATTN_TYPE}`, it can be `'softmax'`, `'fourier'`, `'linear'`, or `'galerkin'`:
```bash
python ex2_memory_profile.py --batch-size 4 --dmodel 128 --attention-type $ATTN_TYPE --subsample-nodes 2 --subsample-attn 7 --num-iter 1000
```
then open up bash and use `nvidia-smi` to check the active Python process's memory.


## Only encoder profiling
The bottleneck of Example 2 and 3 is actually the feature extractor, to profile encoder performance only:
```bash
python encoder_memory_profile.py --seq-len 8192 --batch-size 4 --dmodel 128 --head 1 --num-layers 4 --ndim 2 --num-iter 1000 --attention-type 'galerkin'
```

```bash
python encoder_memory_profile.py --seq-len 8192 --batch-size 4 --dmodel 128 --head 1 --num-layers 4 --ndim 2 --num-iter 1000 --attention-type 'fourier'
```


```bash
python encoder_memory_profile.py --seq-len 8192 --batch-size 4 --dmodel 128 --head 1 --num-layers 4 --ndim 2 --num-iter 1000 --attention-type 'softmax'
```


```bash
python encoder_memory_profile.py --seq-len 8192 --batch-size 4 --dmodel 128 --head 1 --num-layers 4 --ndim 2 --num-iter 1000 --attention-type 'linear'
```

```bash
python encoder_memory_profile.py --seq-len 8192 --batch-size 4 --dmodel 128 --head 1 --num-layers 4 --ndim 2 --num-iter 1 --attention-type 'softmax' 'fourier' 'linear' 'galerkin'
```

Galerkin-type attention has a huge edge over the linear attention for long sequences.