## Image classification

This image classification recipe is heavily based and depends on pytorch-image-models (timm), the awesome tool developed by [Ross Wightman](https://github.com/rwightman).
It supports all data augmentation, datasets and architectures of the original implementation, and was adapted to support the training of PhiNets.

To reproduce our results, you can follow these steps:

1. install PhiNets with `pip install git+https://github.com/fpaissan/micromind`
2. install the additional dependencies for this recipe with `pip install -r extra_requirements.txt`
2. launch the training script on the dataset you want 

### MNIST
```
python classification.py ~/data/mnist -b 128 --dataset torch/mnist --num-classes 10 \
	--model phinet --input-size 1 28 28 --epochs 20 --amp \
	--opt adam --lr 0.01 --weight-decay 0.01 --no-aug \
	--pin-mem --apex-amp --use-multi-epochs-loader --mean 0.1307 --std 0.3081 --dataset-download --log-interval 100 \
	--alpha 0.5 --num_layers 4 --beta 1 --t_zero 6 --experiment mnist
```

### CIFAR-10
```
python classification.py ~/data/cifar10 -b 64 --dataset torch/cifar10 --num-classes 10 \
	--model phinet --input-size 3 160 160 --epochs 100 --amp \
	--opt lamb --sched cosine --lr 0.005 --weight-decay 0.02 --warmup-epochs 10 --warmup-lr 0.008 \
	--hflip 0.5 --aa rand-m3-mstd0.55 --mixup 0.1 --bce-loss \
	--pin-mem --apex-amp --use-multi-epochs-loader --dataset-download --experiment cifar10 \
	--alpha 3 --beta 0.75 --t_zero 6 --num_layers 7
```

### CIFAR-100
```
python classification.py ~/data/cifar100 -b 64 --dataset torch/cifar100 --num-classes 100 \
	--model phinet --input-size 3 160 160 --epochs 100 --amp \
	--opt lamb --sched cosine --lr 0.005 --weight-decay 0.02 --warmup-epochs 10 --warmup-lr 0.008 \
	--hflip 0.5 --aa rand-m3-mstd0.55 --mixup 0.1 --bce-loss \
	--pin-mem --apex-amp --use-multi-epochs-loader --dataset-download --experiment cifar100 \
	--alpha 3 --beta 0.75 --t_zero 6 --num_layers 7
```

In the table is a list of PhiNet's performance on some common image classification benchmarks.

| Dataset | Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| -------- | ------------------ |---------------- | -------------- |
| MNIST | `PhiNet(alpha=0.5, beta=1, t_zero=6, num_layers=4)`   |     98.96%         |      100.00%       |
| CIFAR-10 | `PhiNet(alpha=3, beta=0.75, t_zero=6, num_layers=7)`   |     93.61%         |      99.77%       |
| CIFAR-100 | `PhiNet(alpha=3, beta=0.75, t_zero=6, num_layers=7)`   |     75.56%         |      93.5%       |

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
