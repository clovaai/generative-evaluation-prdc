[![PyPI version](https://badge.fury.io/py/prdc.svg)](https://badge.fury.io/py/prdc)
[![PyPI download month](https://img.shields.io/pypi/dm/prdc.svg)](https://pypi.python.org/pypi/prdc/)
[![PyPI license](https://img.shields.io/pypi/l/prdc.svg)](https://pypi.python.org/pypi/prdc/)

## Reliable Fidelity and Diversity Metrics for Generative Models (ICML 2020)

[Paper: Reliable Fidelity and Diversity Metrics for Generative Models](https://arxiv.org/abs/2002.09797)

Muhammad Ferjad Naeem <sup>1,3*</sup>, Seong Joon Oh<sup>2*</sup>, Yunjey Choi<sup>1</sup>, 
Youngjung Uh<sup>1</sup>, Jaejun Yoo<sup>1,4</sup>  

<sub>**Work done at Clova AI Research**</sub>

<sub>\* Equal contribution</sub>
<sup>1</sup> <sub>Clova AI Research, NAVER Corp.</sub>
<sup>2</sup> <sub>Clova AI Research, LINE Plus Corp.</sub>
<sup>3</sup> <sub>Technische Universit&auml;t M&uuml;nchen</sub>
<sup>4</sup> <sub>EPFL</sub>

Devising indicative evaluation metrics for the image generation task remains an open problem.
The most widely used metric for measuring the similarity between real and generated images has been the Fr&eacute;chet Inception Distance (FID) score. 
Because it does not differentiate the _fidelity_ and _diversity_ aspects of the generated images, recent papers have introduced variants of precision and recall metrics to diagnose those properties separately.
In this paper, we show that even the latest version of the precision and recall (Kynk&auml;&auml;nniemi et al., 2019) metrics are not reliable yet. For example, they fail to detect the match between two identical distributions, they are not robust against outliers, and the evaluation hyperparameters are selected arbitrarily. We propose **density and coverage** metrics that solve the above issues. We analytically and experimentally show that density and coverage provide more interpretable and reliable signals for practitioners than the existing metrics.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=_XwsGkryVpk
" target="_blank"><img src="http://img.youtube.com/vi/_XwsGkryVpk/0.jpg" 
alt="VIDEO" width="700" border="10" /></a>

## Updates

* **1 June 2020**: Paper accepted at ICML 2020.

## 1. Background

### Precision and recall metrics

Precision and recall are defined below:

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\text{precision}:=\frac{1}{M}\sum_{j=1}^{M}1_{Y_j\in\text{manifold}(X_1,\cdots,X_N)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\text{precision}:=\frac{1}{M}\sum_{j=1}^{M}1_{Y_j\in\text{manifold}(X_1,\cdots,X_N)}" title="\text{precision}:=\frac{1}{M}\sum_{j=1}^{M}1_{Y_j\in\text{manifold}(X_1,\cdots,X_N)}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\text{recall}:=\frac{1}{N}\sum_{i=1}^{N}1_{X_i\in\text{manifold}(Y_1,\cdots,Y_M)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\text{recall}:=\frac{1}{N}\sum_{i=1}^{N}1_{X_i\in\text{manifold}(Y_1,\cdots,Y_M)}" title="\text{recall}:=\frac{1}{N}\sum_{i=1}^{N}1_{X_i\in\text{manifold}(Y_1,\cdots,Y_M)}" /></a>

where the manifold is the defined as

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\text{manifold}(X_1,\cdots,X_N):=&space;\bigcup_{i=1}^{N}&space;B(X_i,\text{NND}_k(X_i))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\text{manifold}(X_1,\cdots,X_N):=&space;\bigcup_{i=1}^{N}&space;B(X_i,\text{NND}_k(X_i))" title="\text{manifold}(X_1,\cdots,X_N):= \bigcup_{i=1}^{N} B(X_i,\text{NND}_k(X_i))" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\fn_cm&space;B(x,r)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\fn_cm&space;B(x,r)" title="B(x,r)" /></a> 
is the ball around the point `x` with radius `r`. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\fn_cm&space;\text{NND}_k(X_i)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\fn_cm&space;\text{NND}_k(X_i)" title="\text{NND}_k(X_i)" /></a>
is the distance to the kth-nearest neighbour. 

### Density and coverage metrics

Density and coverage are defined below:

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\text{density}:=\frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}1_{Y_j\in&space;B(X_i,\text{NND}_k(X_i))}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\text{density}:=\frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}1_{Y_j\in&space;B(X_i,\text{NND}_k(X_i))}" title="\text{density}:=\frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}1_{Y_j\in B(X_i,\text{NND}_k(X_i))}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\text{coverage}:=\frac{1}{N}\sum_{i=1}^{N}1_{\exists\text{&space;}j\text{&space;s.t.&space;}&space;Y_j\in&space;B(X_i,\text{NND}_k(X_i))}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\text{coverage}:=\frac{1}{N}\sum_{i=1}^{N}1_{\exists\text{&space;}j\text{&space;s.t.&space;}&space;Y_j\in&space;B(X_i,\text{NND}_k(X_i))}" title="\text{coverage}:=\frac{1}{N}\sum_{i=1}^{N}1_{\exists\text{ }j\text{ s.t. } Y_j\in B(X_i,\text{NND}_k(X_i))}" /></a>


### Why are DC better than PR?

<p align="center">
  <img src="https://github.com/clovaai/prdc/blob/master/figure/p_vs_d.png?raw=true" alt="Precision versus density." width="500"/>
</p>

**Precision versus Density.** 
Because of the real outlier sample, the manifold is overestimated. Generating many fake samples around the real outlier is enough to increase the precision measure. 
The problem of overestimating precision (100%) is resolved using the density estimate (60%). 

<p align="center">
  <img src="https://github.com/clovaai/prdc/blob/master/figure/r_vs_c.png?raw=true" alt="Recall versus coverage." width="600"/>
</p>

**Recall versus Coverage.** 
The real and fake samples are identical across left and right.
Since models often generate many unrealistic yet diverse samples, the fake manifold is often an overestimation of the true fake distribution. 
In the figure above, while the fake samples are generally far from the modes in real samples, the recall measure is rewarded by the fact that real samples are contained in the overestimated fake manifold.


## 2. Usage

### Installation

```bash
pip3 install prdc
```

### Example

Test 10000 real and fake samples form the standard normal distribution N(0,I) in 1000-dimensional Euclidean space.
Set the nearest neighbour `k=5`. We compute precision, recall, density, and coverage estimates below.

```python
import numpy as np
from prdc import compute_prdc


num_real_samples = num_fake_samples = 10000
feature_dim = 1000
nearest_k = 5
real_features = np.random.normal(loc=0.0, scale=1.0,
                                 size=[num_real_samples, feature_dim])

fake_features = np.random.normal(loc=0.0, scale=1.0,
                                 size=[num_fake_samples, feature_dim])

metrics = compute_prdc(real_features=real_features,
                       fake_features=fake_features,
                       nearest_k=nearest_k)

print(metrics)
```
Above test code will result in the following estimates (may fluctuate due to randomness).
```python
{'precision': 0.4772,
 'recall': 0.4705,
 'density': 1.0555,
 'coverage': 0.9735}
```

## 3. Miscellaneous

### References

Kynk&auml;&auml;nniemi et al., 2019. Improved precision and recall metric for assessing generative models. Neurips 2019.

### License

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### Cite this work

```
@article{ferjad2020icml,
  title = {Reliable Fidelity and Diversity Metrics for Generative Models},
  author = {Naeem, Muhammad Ferjad and Oh, Seong Joon and Uh, Youngjung and Choi, Yunjey and Yoo, Jaejun},
  year = {2020},
  booktitle = {International Conference on Machine Learning},
}
```
