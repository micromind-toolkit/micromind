---

<div align="center">    
 
# This repo contains the code for _PhiNets_, a scalable backbone for low-power AI at the edge

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2110.00337)

<!--  
Conference   
-->   
</div>
 
## Description   
This is a first version of the public repo. It will be reorganized with specific instructions to reproduce the results of the paper above (with data download, etc). In the meantime, if you are willing to use the PhiNets in your signal processing pipelines, you can import them in PyTorch from this repo (soon pip!) or use them in keras from [this gist](https://gist.github.com/fpaissan/9157baa162649fba917a211434ae904c).

#### Example: cifar10 
To train PhiNets for cifar10, you can run:

```
python __main__.py cifar10 data
```

### Citation
```
@article{phinet,
Author = {Francesco Paissan and Alberto Ancilotto and Elisabetta Farella},
Title = {{PhiNets: a scalable backbone for low-power AI at the edge}},
Year = {2021},
Eprint = {arXiv:2110.00337},
}
```
