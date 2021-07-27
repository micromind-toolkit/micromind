---

<div align="center">    
 
# PhiNets: a scalable backbone for low-power AI at the edge

[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://dl.acm.org/pb-assets/static_journal_pages/tecs/pdf/CFP_AIatEDGE_TECS-1622578812223.pdf)

<!--  
Conference   
-->   
</div>
 
## Description   
This is a first version of the public repo. It will be reorganized with specific instructions to reproduce the results of the paper above (with data download, etc). In the meantime, if you are willing to use the PhiNets in your signal processing pipelines, you can import them in PyTorch from this repo (soon pip!) or use them in keras from [this gist](https://gist.github.com/fpaissan/9157baa162649fba917a211434ae904c).

_PhiNets_ achieve state-of-the-art results for object detection and tracking in the 1-10MMACC range (typical application range for MCU applications). Check out the paper for more details!

#### Example: cifar10 
To train PhiNets for cifar10, you can run:

```
python __main__.py cifar10 data
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
