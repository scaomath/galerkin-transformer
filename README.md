# Fourier Transformer and Galerkin Transformer: Attention without softmax
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Pytorch 1.8](https://img.shields.io/badge/pytorch-1.8-blue.svg)](https://pytorch.org/)

TL;DR:
The new attention operator is `(QK^T)V` or `Q(K^TV)`, whichever doing matmul gets the layer normalization, i.e., `Q, K` get layer normalized in local attention, as for `K, V` in global attention. No softmax, no layer normalization is applied afterward. This is called a scaling-preserving simple attention. Combining with proper feature extractor and decoder, it is extremely powerful in learning PDE-related operators (energy decay, inverse coefficient identification).


For details please refer to:
```latex
@Misc{Cao:2021transformer,
  author        = {Shuhao Cao},
  title         = {Choose a Transformer: Fourier or Galerkin},
  year          = {2021},
  archiveprefix = {arXiv},
  eprint        = {},
  primaryclass  = {cs.CL},
}
```


# Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

# Data
The data is courtesy of [Zongyi Li (Caltech)](https://github.com/zongyi-li/fourier_neural_operator)  under the MIT license. Download the data from [here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing), and unzip the Burgers and Darcy flow problem files to the `./data` for 
>`burgers_data_R10.mat`
<br>`piececonst_r421_N1024_smooth1.mat`
<br>`piececonst_r421_N1024_smooth2.mat`.

The code has a semi env variable `DATA_PATH` set in [`utils_ft.py`](./libs/utils_ft.py).

# Examples
All examples are learning PDE-related operators. The setting can be found in [`config.yml`](./config.yml). By default the evaluation is performed on the last 100 samples in the test dataset. All trainers are using the [`1cycle` scheduler](https://arxiv.org/abs/1708.07120) in [PyTorch](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.OneCycleLR.html) for 100 epochs. Every example has a `--seed {$SEED}` argument and the default seed is 1127802. Since [`nn.functional.interpolate`](https://pytorch.org/docs/master/generated/torch.nn.functional.interpolate.html) is used in 2D examples, a fixed seed may still yield different results each training cycle on GPU according to PyTorch documents, but we have verified that the variance is negligible. Some example set-ups are as follows, to fully reproducing our result, please refer to [`training.md`](./training.md).

## Example 1: Burgers equation
The baseline benchmark [`ex1_burgers.py`](./ex1_burgers.py): evaluation relative error is about `1e-3` with a simple pointwise forward expansion feature extractor. The input is the initial condition of a viscous Burgers' equation on a discrete grid, the output is an approximation to the solution marched to time $1$. The initial data are generating using a GRF and the data in the validation set are not in the train set.

Default benchmark on a 2048 grid using a Fourier Transformer, with 4 Fourier-type attention layers as the encoder and 2 spectral convolution layers from [Li et al 2020](https://github.com/zongyi-li/fourier_neural_operator) as the decoder:
```bash
python ex1_burgers.py
```
No subsampling (8192 grid), Galerkin-type attention, adding a diagonal matrix to the Xavier initializations of the `W^Q, W^K, W^V` matrices (about 30%-1000% better than those without depending on other settings).
```bash
python ex1_burgers.py --subsample 1\
                      --attention-type 'galerkin'\
                      --xavier-init 0.01 --diag-weight 0.01
```
Using standard softmax normalization `Softmax(QK^T/sqrt{d})V`, conventional layer normalization application scheme in attention layers
```bash
python ex1_burgers.py --attention-type 'softmax'\
                      --reg-layernorm\
                      --xavier-init 1.0 --diag-weight 0.0
```

## Example 2: Interface Darcy flow
The baseline benchmark [`ex2_darcy.py`](./ex2_darcy.py): evaluation relative error is about `8e-3` to `1e-2` with a 3-level interpolation-based CNN (CiNN) feature extractor. The coarse grid latent representation is sent to attention layers The operator input is discontinuous coefficient with a random interface sampled at a discrete grid, the output is a finite difference approximation to the solution restricted to the sampled grid from a fine `421x421` grid. The coefficient in the validation set are not in the train set.

Default benchmark on a 141x141 grid using the Galerkin Transformer, 6 Galerkin-type attention layers with `d_model=128` and `nhead=4` as the encoder, and 2 spectral conv layers from [Li et al 2020](https://github.com/zongyi-li/fourier_neural_operator) as the decoder. There is a small dropout `5e-2` in the attention layer as well as in the feature extraction layer:
```bash
python ex2_darcy.py --reg-layernorm
```
For a smaller memory GPU or CPU, please use the 85x85 grid fine, 29x29 coarse grid example:
```bash
python ex2_darcy.py --subsample-attn 15\
                    --subsample-nodes 5\
                    --attention-type 'galerkin'\
                    --reg-layernorm\
                    --xavier-init 0.01 --diag-weight 0.01
```
## Example 3: Inverse interface coefficient identification for Darcy flow
The baseline benchmark [`ex3_darcy_inv.py`](./ex3_darcy_inv.py): an inverse coefficient identification problem based on the same dataset used in Example 2. However, in this example, the input and the target are reversed, i.e., the target is the interface coefficient with a random geometry, and the input is the finite difference approximation to the PDE problem, together with an optional noise added to the input to simulate measurement errors. Due to a limit of interpolation operator having no approximation property to nonsmooth functions, the coefficient cannot be resolved at the resolution, the target is sampled at a lower resolution than the input. Evaluation relative error is about `1.5e-2` to `2e-2` without noise, `3e-2` with 1% noise, and `7e-2` to `8e-2` with 10% noise. The main source of the error stems from the magnitudes of the coefficient, the attention-based learner can capture the random interface geometry pretty well.

Default benchmark is on a 211x211 fine grid input and a 71x71 coarse grid coefficient output. The model is the Galerkin Transformer with 6 stacked Galerkin-type attention layers (`d_model=192`, `nhead=4`) with a simple pointwise feed-forward neural network to map the attention output back the desired dimension. There is a small dropout in every key components of the network (`5e-2`). The noise is added to the normalized input, so 0.01 noise means 1%, and 0.1 means 10%.
```bash
python ex3_darcy_inv.py --noise 0.01
```
For a smaller memory GPU, please use the 141x141 grid fine, 36x36 coarse grid, and avoid using the local attention `fourier` or `softmax` in the `--attention-type` switch:
```bash
python ex3_darcy_inv.py --subsample-attn 12\
                        --subsample-nodes 3\
                        --attention-type 'galerkin'\
                        --xavier-init 0.01 --diag-weight 0.01
```

# Memory and speed profiling using `autograd.profiler`
Using CUDA, Fourier Transformer features an over 40% reduction in `self_cuda_memory_usage` versus the standard softmax normalized transformers, and Galerkin Transformer's the backpropagation speed has a 20% to 100% increase over the standard linearized transformers. If no GPU is available please enable the `--no-cuda` switch.

Example 1 memory profile of a model with 96 hidden dimension with an input sequence length 8192. Compare the memory usage of the Fourier transformer with the one with softmax
```bash
python ex1_memory_profile.py --batch-size 4 --seq-len 8192 --dmodel 96 --attention-type 'softmax' 'fourier'
```
Compare the backpropagation time usage of the Galerkin transformer versus the same net, but with Galerkin-type simple attention replaced by the standard linearized attention. 
```bash
python ex1_memory_profile.py --batch-size 4 --seq-len 8192 --dmodel 96 --num-iter 10 --attention-type 'linear' 'galerkin'
```

Encoder layer wrapper profiling: profile a wrapper with 10 layers of encoder in a model for operators defined for functions whose domain is isomorphic to a 2D Euclidean space.
```bash
python encoder_memory_profile.py --batch-size 4 --dmodel 96 --num-layers 10 -ndim 2
```
Please refer to [`training.md`](./training.md) for more detailed profiling in each example.


# License
This software is distributed with the MIT license which translates roughly that you can use it however you want and for whatever reason you want. All the
information regarding support, copyright and the license can be found in the [LICENSE](./LICENSE) file.

# Acknowledgement
This part is omitted due to the NeurIPS's anonymous policy.

The hardware to perform this work is provided by Andromeda Saving Fund. This work was supported in part by the National Science Foundation under grants DMS-1913080 and no additional revenues are related to this work. We would like to thank [Dr. Long Chen (Univ of California Irvine)](github.com/lyc102) for the inspiration of and encouragement on the initial conceiving of this paper, as well as numerous constructive advices on revising this paper, not mentioning his persistent dedication of [making publicly available tutorials](https://www.math.uci.edu/~chenlong/226/Ch3FEMCode.pdf) on writing [beautiful vectorized code](https://github.com/lyc102/ifem). We would like to thank Dr. Ari Stern (Washington Univ in St. Louis) for the help on the relocation during the COVID-19 pandemic. We would like to thank Dr. Ruchi Guo (Univ of California Irvine) and Dr. Yuanzhe Xi (Emory) for the invaluable feedbacks on the choice of the numerical experiments. We would like to thank the Kaggle community, including but not limited to Jean-François Puget ([Uncle CPMP@Kaggle](https://www.kaggle.com/cpmpml)) and Murakami Akira ([mrkmakr@Kaggle](https://www.kaggle.com/mrkmakr)) for sharing a simple [Graph Transformer](https://www.kaggle.com/cpmpml/graph-transfomer). 
We would like to thank daslab@Stanford, OpenVaccine, and Eterna for hosting the COVID-19 mRNA Vaccine competition and Deng Lab (Univ of Georgia) for collaborating in this competition. We would like to thank CHAMPS (Chemistry and Mathematics in Phase Space) for hosting the J-coupling quantum chemistry competition and Corey Levinson ([returnofsputnik@Kaggle](https://www.kaggle.com/returnofsputnik), Eligo Energy, LLC) for collaborating in this competition. We would like to thank [Zongyi Li (Caltech)](https://github.com/zongyi-li) for sharing some early dev code in the updated PyTorch `torch.fft` interface. We would like to thank Ziteng Pang (Univ of Michigan) to update us with various references on Transformers. We would like to thank [Joel Schlosser](https://github.com/jbschlosser) to incorporate our change to the PyTorch `transformer` submodule to simplify our testing pipeline. We would be grateful to the PyTorch community for selflessly code sharing, including Phil Wang([lucidrains@github](https://github.com/lucidrains)) and [Harvard NLP group Klein et al. (2017)](https://nlp.seas.harvard.edu/2018/04/03/attention.html). We would like to thank the chebfun Driscoll et al. (2014) for integrating powerful tools into a simple interface to solve PDEs. We would like to thank [Dr. Yannic Kilcher](https://www.youtube.com/c/YannicKilcher/about) and [Dr. Hung-yi Lee (National Taiwan Univ)](https://www.youtube.com/c/HungyiLeeNTU) for frequently covering the newest research on Transformers in video formats. We would also like to thank the Python community (Van Rossum and Drake (2009); Oliphant (2007)) for sharing and developing the tools that enabled this work, including Pytorch Paszke et al.(2017), NumPy Harris et al. (2020), SciPy Virtanen et al. (2020), Seaborn Waskom (2021), Plotly Inc. (2015) and Matplotlib Hunter (2007).
