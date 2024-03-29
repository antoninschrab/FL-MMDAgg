# FL-MMDAgg

We evaluate the performance our MMDAgg test ([paper](https://arxiv.org/abs/2110.15073), [code](https://github.com/antoninschrab/mmdagg-paper)) on the Failing Loudly benchmark ([paper](https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html), [code](https://github.com/steverab/failing-loudly/)) which has also been considered by Kübler et al. ([paper](https://arxiv.org/abs/2206.08843), [code](https://github.com/jmkuebler/autoML-TST-paper)). The code in this repository is based on the two aforementionned repositories which are both under the MIT License.

We run experiments using both the old version [mmdagg_old.py](mmdagg_old.py) and the current version [mmdagg.py](mmdagg.py) of MMDAgg. 

Our MMDAgg test can be run in practice using the `mmdagg` package implemented the [mmdagg repository](https://github.com/antoninschrab/mmdagg/), which contains both a Numpy version and a Jax version.

## Results (table)

First, we report the MMDAgg results for the Laplace and Gaussian kernels using the old version of MMDAgg with a collection of bandwidths consisting of $2^\ell \lambda_{med}$ for $\ell=10,\dots,20$ where $\lambda_{med}$ is the median bandwidth. This collection of bandwidths has been considered for the MMD test splitting the data in the MNIST experiment of [MMD Aggregated Two-Sample Test](https://arxiv.org/abs/2110.15073) in Section 5.4.

| Sample size | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 10000 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| MMDAgg Laplace | 0.20 | 0.28 | 0.40 | 0.43 | 0.46 | 0.52 | 0.58 | 0.79 |
| MMDAgg Gaussian | 0.15 | 0.23 | 0.33 | 0.35 | 0.38 | 0.44 | 0.48 | 0.69 |

As observed in the MNIST experiment of [MMD Aggregated Two-Sample Test](https://arxiv.org/abs/2110.15073) in Figure 5, MMDAgg Laplace outperforms MMDAgg Gaussian.

We now report results using the current version of MMDAgg, which is introduced in Section 5.2 of [MMD Aggregated Two-Sample Test](https://arxiv.org/abs/2110.15073) and referred to as $\textrm{MMDAgg}^\star$ in the paper. This test uses an adaptive parameter-free collection of bandwidths. The test MMDAgg All aggregates 12 types of kernels, each with 10 bandwidths, details are provided in Section 5.4 of [MMD Aggregated Two-Sample Test](https://arxiv.org/abs/2110.15073).

| Sample size | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 10000 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| MMDAgg Laplace | 0.21 | 0.29 | 0.40 | 0.44 | 0.47 | 0.56 | 0.67 | 0.83 |
| MMDAgg Gaussian | 0.19 | 0.26 | 0.34 | 0.42 | 0.42 | 0.51 | 0.62 | 0.75 |
| MMDAgg Laplace & Gaussian | 0.21 | 0.27 | 0.37 | 0.43 | 0.45 | 0.54 | 0.65 | 0.80 |
| MMDAgg All | 0.21 | 0.27 | 0.37 | 0.43 | 0.46 | 0.55 | 0.66 | 0.80 |

We observe that this parameter-free version of MMDAgg obtains higher power than the one presented above with the collection consisting of $2^\ell \lambda_{med}$ for $\ell=10,\dots,20$ where $\lambda_{med}$ is the median bandwidth.
The test retains high power when aggregating over many more kernels (i.e. MMDAgg All).

## Datasets

The adversarial datasets can either be generated by running 
```
python generate_adv_samples.py
```
which saves them in the `datasets` directory,
or they can be directly downloaded from the [failing-loudly](https://github.com/steverab/failing-loudly/tree/42afd118237ded54c6ebef4a3417d8c1db44f76d/datasets) repository. 

## Experiments

The environment is the same as the one considered in the [autoML-TST-paper](https://github.com/jmkuebler/autoML-TST-paper) repository, it can be installed by following their instructions.

The experiments can be run by first editing the parameters (choice of version and of kernel for MMDAgg) at the beginning of the [pipeline.py](pipeline.py) and [shift_tester.py](shift_tester.py) files, and then executing
```
bash script.sh
```
The results are saved in the [paper_results/](paper_results/) directory.
The experiments consist of 'embarrassingly parallel for loops' which can be computed efficiently using parallel computing libraries such as `joblib` or `dask`.

The test power for MMDAgg for those experiments can then be obtained by running
```
python results.py
```
The results are presented in the tables above.

## License

MIT License (see [LICENSE.md](LICENSE.md))

