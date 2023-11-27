accelerate launch train.py cfg/phinet.py -b 64 --dataset torch/cifar10 --num-classes 10 \
	--hflip 0.5 --aa rand-m3-mstd0.55 --mixup 0.1 --experiment_name cifar10_bce --bce_loss True 
