accelerate launch train.py cfg/phinet.py -b 64 --dataset torch/cifar10 --num-classes 10 \
	--model phinet -epochs 100 --amp \
	-lr 0.005 --weight-decay 0.02 \
	--experiment_name cifar10 \
	--alpha 3 --beta 0.75 --t_zero 6 --num_layers 7
	# --hflip 0.5 # --aa rand-m3-mstd0.55 --mixup 0.1 --bce-loss \
