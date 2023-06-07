## Object detection

This object detection recipe is heavily based and depends on YOLO v8, developed by [Ultralytics](https://github.com/ultralytics/ultralytics)
It supports training and inference, for now tested on the COCO dataset.

To reproduce our results, you can follow these steps:

1. install PhiNets with `pip install git+https://github.com/fpaissan/micromind`
2. install the additional dependencies for this recipe with `pip install -r extra_requirements.txt`
3. launch the training script on the dataset you want

### COCO
```
python classification.py ~/data/mnist -b 128 --dataset torch/mnist --num-classes 10 \
	--model phinet --input-size 1 28 28 --epochs 20 --amp \
	--opt adam --lr 0.01 --weight-decay 0.01 --no-aug \
	--pin-mem --apex-amp --use-multi-epochs-loader --mean 0.1307 --std 0.3081 --dataset-download --log-interval 100 \
	--alpha 0.5 --num_layers 4 --beta 1 --t_zero 6 --experiment mnist
```

In the table is a list of PhiNet's performance on some common image classification benchmarks.

| Dataset | Model name         | mAP50   |
| -------- | ------------------ |---------------- |
| COCO	  | `PhiNet(alpha=0.5, beta=1, t_zero=6, num_layers=4)`   |     98.96%         |
| CIFAR-10 | `PhiNet(alpha=3, beta=0.75, t_zero=6, num_layers=7)`   |     93.61%         |
| CIFAR-100 | `PhiNet(alpha=3, beta=0.75, t_zero=6, num_layers=7)`   |     75.56%         |


### Cite PhiNets
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