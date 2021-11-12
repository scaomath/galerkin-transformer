# [NeurIPS 2021] Galerkin Transformer: linear attention without softmax
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Pytorch 1.9](https://img.shields.io/badge/pytorch-1.9-blue.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2105.14995-b31b1b.svg)](https://arxiv.org/abs/2105.14995)
[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/scaomath/galerkin-transformer)


# Summary
- A non-numerical analyst oriented explanation on Toward Data Science about the [Galerkin Transformer](https://towardsdatascience.com/galerkin-transformer-a-one-shot-experiment-at-neurips-2021-96efcbaefd3e)
-  [The post on my blog](https://scaomath.github.io/blog/galerkin-transformer/), has much more details on the math of how to bridge the attention operator (a nonlinear operator)'s approximation capacity with a linear operator (Petrov-Galerkin projection).
- The post on my mentor's WeChat blog CAM传习录 (in Chinese): [Galerkin Transformer: 初学者的进击](https://mp.weixin.qq.com/s?__biz=MzUxNzk0NjExOA==&mid=2247487695&idx=1&sn=be0c364d5d85ca83ee27f4425d0a38c2)

- For how to train our models please refer to [the training instructions under the `/examples` folder](./examples/).

- If just wanting to see what is it like for the models to perform on the unseen test set, please refer to [evaluation](#evaluation-notebooks).

## Introduction 
The new simple attention operator (for the encoder) is simply `Q(K^TV)` (Galerkin), or the quadratic complexity one `(QK^T)V` (Fourier). 
- No softmax, or the approximation thereof, at all.
- Whichever two latent representations doing `matmul` get the layer normalization, similar to Gram-Schmidt process where we have to divide the basis's norm squared. `Q, K` get layer normalized in the Fourier-type attention (every position attends with every other), as for `K, V` in the Galerkin-type attention (every basis attends with every other basis). No layer normalization is applied afterward. 
- Some other components are tweaked according to our Hilbertian interpretation of attention.

For the full operator learner, the feature extractor is a simple linear layer or an interpolation-based CNN, the decoder is the spectral convolution real parameter re-implementation from the best operator learner to-date Fourier Neural Operator (FNO) in [*Li et al 2020*](https://github.com/zongyi-li/fourier_neural_operator) if the target is smooth, or just a pointwise FFN if otherwise. The resulting network is extremely powerful in learning PDE-related operators (energy decay, inverse coefficient identification).

## Hilbertian framework to analyze a linear attention variant
Even though everyone is Transformer'ing, the mathematics behind the attention mechanism is not well understood. We have also shown that the Galerkin-type attention (a linear attention without softmax) has an approximation capacity on par with a Petrov-Galerkin projection under a Hilbertian setup. We use a method commonly known as ''mixed method'' in the finite element analysis community that is used to solve fluid/electromagnetics problems. Unlike finite element methods, in an attention-based operator learner the approximation is not discretization-tied, in that:

1. The latent representation is interpreted "column-wise" (each column represents a basis), opposed to the conventional "row-wise"/ "position-wise"/"word-wise" interpretation of attention in NLP.
2. The dimensions of the approximation spaces are not tied to the geometry as in the traditional finite element analysis (or finite difference, spectral methods, radial basis, etc.);
3. The approximation spaces are being dynamically updated by the nonlinear universal approximator due to the presence of the positional encodings, which determines the topology of the approximation space.

## Interpretation of the attention mechanism
1. Approximation capacity: an incoming "query" is a function in some Hilbert space that comes to ask us to find its best representation in the latent space. To deliver the best approximator in "value" (trial function space), the "key" space (test function space) has to be big enough so that for every value there is a key to unlock it.
2. Translation capacity: the attention is capable to find latent representations to minimize a functional norm that measures the distance between the input (query) and the target (values). An ideal operator learner is learning some nonlinear perturbations of the subspaces on which the input (query) and the target (values) are "close", and this closeness is measured by how they respond to a dynamically changing set of test basis (keys).

For details please refer to: [https://arxiv.org/abs/2105.14995](https://arxiv.org/abs/2105.14995)
```bibtex
@inproceedings{Cao2021transformer,
  author        = {Shuhao Cao},
  title         = {Choose a Transformer: {F}ourier or {G}alerkin},
  booktitle     = {Thirty-Fifth Conference on Neural Information Processing Systems (NeurIPS 2021)},
  year          = {2021},
  eprint        = {arXiv: 2105.14995},
  primaryclass  = {cs.CL},
  url={https://openreview.net/forum?id=ssohLcmn4-r},
}
```

# Install

## Requirements
(Updated Jun 17 2021) `PyTorch` requirement updated to `1.9.0` as the introduction of the [`batch_first` argument](https://github.com/pytorch/pytorch/pull/55285) will conform with our pipeline.

This package can be cloned locally and used with the following requirements:

```bash
git clone https://github.com/scaomath/galerkin-transformer.git
cd galerkin-transformer
python3 -m pip install -r requirements.txt
```

```sh
seaborn==0.11.1
torchinfo==0.0.8
numpy==1.20.2
torch==1.9.0
plotly==4.14.3
scipy==1.6.2
psutil==5.8.0
matplotlib==3.3.4
tqdm==4.56.0
PyYAML==5.4.1
```

If interactive mode is to be used, please install
```
jupyterthemes==0.20.0
ipython==7.23.1
```

## Installing using pip

This package can be installed using pip.

```bash
python3 -m pip install galerkin-transformer
```

Example usage of the Simple Fourier/Galerkin Transformer encoder layers:

```python
from galerkin_transformer.model import *

encoder_layer = FourierTransformerEncoderLayer(
                 d_model=128,
                 pos_dim=1,
                 n_head=4,
                 dim_feedforward=512,
                 attention_type='galerkin',
                 layer_norm=False,
                 attn_norm=True,
                 norm_type='layer',
                 dropout=0.05)
encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(6)])
x = torch.randn(8, 8192, 128) # embedding
pos = torch.arange(0, 8192).unsqueeze(-1) # Euclidean coordinates
pos = pos.repeat(8, 1, 1)
for layer in encoder_layers:
    x = layer(x, pos)
```

# Data
The data is courtesy of [Zongyi Li (Caltech)](https://github.com/zongyi-li/fourier_neural_operator)  under the MIT license. Download the following data from [here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing):
>`burgers_data_R10.mat`
<br>`piececonst_r421_N1024_smooth1.mat`
<br>`piececonst_r421_N1024_smooth2.mat`.

The repo has a semi env variable `$DATA_PATH` set in [`utils_ft.py`](./libs/utils_ft.py), if you have a global system environ variable name `DATA_PATH`, then please put the data in that folder. Otherwise, please unzip the Burgers and Darcy flow problem files to the `./data` folder. 

# Examples
All examples are learning PDE-related operators. The setting can be found in [`config.yml`](./config.yml). To fully reproducing our result, please refer to [the training scripts](./examples/README.md) in the [Examples](./examples/) for all the possible args. 

The memory and speed profiling scripts using `autograd.profiler` can be found in [Examples](./examples/) folder as well.


# Evaluation notebooks
Please download the pretrained model's `.pt` files from Releases and put them in the `./models` folder.
- [Example 1](./eval/ex1_burgers_eval.ipynb)
- [Example 2](./eval/ex2_darcy_eval.ipynb)
- [Example 3](./eval/ex3_darcy_inv_eval.ipynb)

# License
This software is distributed with the MIT license which translates roughly that you can use it however you want and for whatever reason you want. All the
information regarding support, copyright and the license can be found in the [LICENSE](./LICENSE) file.

# Acknowledgement
The hardware to perform this work is provided by Andromeda Saving Fund. The first author was supported in part by the National Science Foundation
under grants DMS-1913080 and DMS-2136075. No additional revenues are related to this work. We would like to thank the anonymous reviewers and the area chair in NeurIPS 2021 for [the suggestions on improving this paper](https://openreview.net/forum?id=ssohLcmn4-r). We would like to thank [Dr. Long Chen (Univ of California Irvine)](https://github.com/lyc102) for the inspiration of and encouragement on the initial conceiving of this paper, as well as numerous constructive advices on revising this paper, not mentioning his persistent dedication of [making publicly available tutorials](https://lyc102.github.io/ifem/) on writing [beautiful vectorized code](https://github.com/lyc102/ifem). We would like to thank Dr. Ari Stern (Washington Univ in St. Louis) for the help on the relocation during the COVID-19 pandemic. We would like to thank Dr. Likai Chen (Washington Univ in St. Louis) for the invitation to [the Stats and Data Sci seminar at WashU](https://math.wustl.edu/events/statistics-and-data-science-seminar-transformer-dissection-amateur-applied-mathematician) that resulted the reboot of this study. We would like to thank Dr. Ruchi Guo (Univ of California Irvine) and Dr. Yuanzhe Xi (Emory) for the invaluable feedbacks on the choice of the numerical experiments. We would like to thank the Kaggle community, including but not limited to Jean-François Puget ([Uncle CPMP@Kaggle](https://www.kaggle.com/cpmpml)) for sharing a simple [Graph Transformer](https://www.kaggle.com/cpmpml/graph-transfomer) in Tensorflow, Murakami Akira ([mrkmakr@Kaggle](https://www.kaggle.com/mrkmakr)) for sharing a [Graph Transformer with a CNN feature extractor in Tensorflow](https://www.kaggle.com/mrkmakr/covid-ae-pretrain-gnn-attn-cnn), and Cher Keng Heng ([hengck23@Kaggle](https://www.kaggle.com/hengck23)) for sharing a [Graph Transformer in PyTorch](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183518).
We would like to thank daslab@Stanford, OpenVaccine, and Eterna for hosting the COVID-19 mRNA Vaccine competition and Deng Lab (Univ of Georgia) for collaborating in this competition. We would like to thank CHAMPS (Chemistry and Mathematics in Phase Space) for hosting the J-coupling quantum chemistry competition and Corey Levinson ([returnofsputnik@Kaggle](https://www.kaggle.com/returnofsputnik), Eligo Energy, LLC) for collaborating in this competition. We would like to thank [Zongyi Li (Caltech)](https://github.com/zongyi-li) for sharing some early dev code in the updated PyTorch `torch.fft` interface. We would like to thank Ziteng Pang (Univ of Michigan) and Tianyang Lin (Fudan Univ) to update us with various references on Transformers. We would like to thank [Joel Schlosser](https://github.com/jbschlosser) to incorporate our change to the PyTorch `transformer` submodule to simplify our testing pipeline. We would be grateful to the PyTorch community for selflessly code sharing, including Phil Wang([lucidrains@github](https://github.com/lucidrains)) and [Harvard NLP group Klein et al. (2017)](https://nlp.seas.harvard.edu/2018/04/03/attention.html). We would like to thank the chebfun Driscoll et al. (2014) for integrating powerful tools into a simple interface to solve PDEs. We would like to thank [Dr. Yannic Kilcher](https://www.youtube.com/c/YannicKilcher/about) and [Dr. Hung-yi Lee (National Taiwan Univ)](https://www.youtube.com/c/HungyiLeeNTU) for frequently covering the newest research on Transformers in video formats. We would also like to thank the Python community (Van Rossum and Drake (2009); Oliphant (2007)) for sharing and developing the tools that enabled this work, including Pytorch Paszke et al.(2017), NumPy Harris et al. (2020), SciPy Virtanen et al. (2020), Seaborn Waskom (2021), Plotly Inc. (2015), Matplotlib Hunter (2007), and the Python team for Visual Studio Code. We would like to thank [`draw.io`](https://github.com/jgraph/drawio)  for providing an easy and powerful interface for producing vector format diagrams. For details please refer to the documents of every function that is not built from the ground up in our open-source software library.
