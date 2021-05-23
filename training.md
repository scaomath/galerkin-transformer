# Training details for each model


## Example 1: viscous Burgers' equation
On the finest grid, `n=8192`. Note that softmax normalization will result diverging training depending on seed. 

Best Fourier Transformer model:
```bash
python ex1_burgers.py --subsample 1\
                      --attention-type 'fourier'\
                      --xavier-init 0.001 --diag-weight 0.01\
                      --batch-size 4
```

Best Galerkin Transformer model:
```bash
python ex1_burgers.py --subsample 1\
                      --attention-type 'galerkin'\
                      --xavier-init 0.01 --diag-weight 0.01\
                      --batch-size 4
```