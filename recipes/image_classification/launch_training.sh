# MNIST training
python classification.py ~/data/ -b 512 --dataset torch/mnist --num-classes 10 \
	--model phinet --input-size 1 28 28 --epochs 20 --amp \
	--opt adam --lr 0.01 --weight-decay 0.01 --no-aug \
	--pin-mem --apex-amp --use-multi-epochs-loader --mean 0.1307 --std 0.3081 --log-interval 100 \
	--alpha 0.5 --num_layers 4 --beta 1 --t_zero 6 --experiment mnist

# CIFAR-10 training
python classification.py ~/data/cifar10 -b 256 --dataset torch/cifar10 --num-classes 10 \
	--model phinet --input-size 3 160 160 --epochs 100 --amp \
	--opt lamb --sched cosine --lr 0.005 --weight-decay 0.02 --warmup-epochs 10 --warmup-lr 0.008 \
	--hflip 0.5 --aa rand-m3-mstd0.55 --mixup 0.1 --bce-loss \
	--pin-mem --apex-amp --use-multi-epochs-loader --dataset-download --experiment cifar10 \
	--alpha 3 --beta 0.75 --t_zero 6 --num_layers 7

# CIFAR-100 training
python classification.py ~/data/cifar100 -b 200 --dataset torch/cifar100 --num-classes 10 \
	--model phinet --input-size 3 160 160 --epochs 100 --amp \
	--opt lamb --sched cosine --lr 0.005 --weight-decay 0.02 --warmup-epochs 10 --warmup-lr 0.008 \
	--hflip 0.5 --aa rand-m3-mstd0.55 --mixup 0.1 --bce-loss \
	--pin-mem --apex-amp --use-multi-epochs-loader --dataset-download --experiment cifar100 \
	--alpha 3 --beta 0.75 --t_zero 6 --num_layers 7 --num-classes 100
