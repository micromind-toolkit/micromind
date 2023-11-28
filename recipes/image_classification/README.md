## Image classification

**Disclaimer**: we will shortly releease HuggingFace checkpoints for ImageNet, CIFAR-100, and CIFAR-10 for both PhiNet and XiNet.

This image classification recipe uses the PyTorch image models library (`timm`) to augment the data. It supports most data augmentation strategies, and datasets of the original implementation. However, it is implemented using `micromind` and thus, it exploits all the exportability and functionalities of the library.

To reproduce our results, you can follow these steps:

1. install PhiNets with `pip install git+https://github.com/fpaissan/micromind`
2. install the additional dependencies for this recipe with `pip install -r extra_requirements.txt`
3. start a training!

### Training

The experiment's configuration is stored inside the files in the `cfg` folder. They can be overridden simply from the command line by providing a new value. For example, if you want to start a training on CIFAR-10, you just need to execute the following command:
```
python train.py cfg/phinet.py
```

For CIFAR-100 instead, you can use:
```
python train.py cfg/phinet.py --dataset torch/cifar100 --data_dir data/cifar100
```

### Inference
In order to export the model and/or run an inference using PyTorch, you can pass an image and the path to a pretrained model to the inference script.
For this, you can use this command:
```
python inference.py cfg/phinet.py IMG_PATH --ckpt_pretrained MODEL_PATH
```

This will print the predicted output, and save an ONNX model in `model.onnx`.


The script will also save an ONNX model at the end of the training. To export the checkpoint in a different format, please read [our documentation](https://micromind-toolkit.github.io/docs/).

#### Referencing PhiNet
If you use PhiNet or `micromind`, please cite our work:
```
@article{10.1145/3510832,
	author = {Paissan, Francesco and Ancilotto, Alberto and Farella, Elisabetta},
	title = {PhiNets: A Scalable Backbone for Low-Power AI at the Edge},
	year = {2022},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3510832},
	doi = {10.1145/3510832},
	journal = {ACM Trans. Embed. Comput. Syst.},
}
```
