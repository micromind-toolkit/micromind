python classification.py ~/data/ -b 512 --dataset torch/cifar10 --num-classes 10 \
	--model phinet --input-size 3 224 224 --epochs 600 --amp \
	--opt lamb --sched cosine --lr 0.005 --weight-decay 0.01 --warmup-epochs 10 --warmup-lr 0.01 \
	--hflip 0.5 --aa rand-m3-mstd0.55 --mixup 0.2 --cutmix 1.0 \
	--bce-loss --smoothing 0.1 --dataset-download
