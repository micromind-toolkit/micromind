# phinet_pl
PhiNet module for PyTorch and PyTorch Lightning use.

Code from the paper "PhiNets: achieving real-time Multi-Object Tracking on tiny architectures" - F. Paissan, A. Ancilotto, E. Farella.

# Image Processing on tiny architectures

Code for "_PhiNets: achieving real-time Multi-Object Tracking on tiny architectures_" - F. Paissan, A. Ancilotto, E. Farella.
This is a first version of the repo. It will be reorganized with specific instructions to reproduce the results of the paper above (with data download, etc).

In the meantime, if you are willing to use the PhiNets in your signal processing pipelines, read below where to find models to import.

### Keras implementation

You can generate and import PhiNets in keras with the source file [phi_net_source.py](https://github.com/fpaissan/phinet-mot/blob/v1/model/phi_net_source.py). The naming convention for the parameters is the same as in the paper.

### Torch implementation

For a PyTorch implementation refer to the [torch repo](https://github.com/fpaissan/phinet_pl).
PhiNets will soon be available for torch hub and pip install! Stay tuned.

## PhiNets results

_PhiNets_ achieve state-of-the-art results for object detection and tracking in the 1-10MMACC range (typical application range for MCU applications).

Below you can see the performance for object detection on COCO and VOC2012 and tracking performance on MOT15.

![Classification performance on COCO and VOC2012](https://github.com/fpaissan/phinet-mot/blob/v1/figs/macc_vs_map.png)

![Tracking performance on MOT15](https://github.com/fpaissan/phinet-mot/blob/v1/figs/mot15_tracking.png)
