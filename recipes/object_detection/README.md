## Object Detection using YOLO

**[16 Jan 2024]** Added optimized YOLO neck, using XiConv. Fixed compatibility with ultralytics weights.<br />
**[17 Dec 2023]** Add VOC dataset, selective head option, and instructions for dataset download.<br />
**[1 Dec 2023]** Fix DDP handling and computational graph.

**Disclaimer**: we will shortly release HuggingFace checkpoints for COCO and VOC for both PhiNet and XiNet.

In an attempt to showcase the simplicity of the YOLO object detection pipeline, we propose our implementation
free of the many abstraction layers of current state-of-the-art implementations. In fact, our implementation targets having not more than two abstraction layers, so that changes and improvements are transparent and reproducibile.

This recipe uses some components from state-of-the-art object detection pipelines (via ultralytics), and supports distributed training.

To reproduce our results, you can follow these steps:

1. install `micromind` with `pip install git+https://github.com/fpaissan/micromind`
2. install the additional dependencies for this recipe with `pip install -r extra_requirements.txt`

**Note**: before training, do not start the process using DDP, if you need to download the dataset.

### Training

The experiment's configuration is stored inside the files in the `cfg` folder. They can be overridden simply from the command line by providing a new value. To start a training on COCO using YOLOPhiNet, you can use:
```
python train.py cfg/yolo_phinet.py
```

If you want to scale the input resolution, you can simply override the argument from the CLI, as in:
```
python train.py cfg/yolo_phinet.py --input_shape 3,96,96
```

### Inference
In order to export the model and/or run an inference using PyTorch, you can pass an image and the path to a pretrained model to the inference script.
For this, you can use this command:
```
python inference.py cfg/yolo_phinet.py IMG_PATH --ckpt_pretrained MODEL_PATH
```

This will print the predicted output, and save an ONNX model in `model.onnx`.

#### Referencing PhiNet
If you use PhiNet or `micromind`, please cite our work:
```
@article{Paissan_2022_TECS,
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

#### Referencing XiNet
If you use XiNet or `micromind`, please cite our work:
```
@InProceedings{Ancilotto_2023_ICCV,
    author    = {Ancilotto, Alberto and Paissan, Francesco and Farella, Elisabetta},
    title     = {XiNet: Efficient Neural Networks for tinyML},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16968-16977}
}
```
