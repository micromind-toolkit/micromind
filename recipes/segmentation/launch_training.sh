CUDA_VISIBLE_DEVICES=0 python detection.py ~/data/mnist -b 128 --dataset torch/mnist --num-classes 10 \
	--model phinet --input-size 1 28 28 --epochs 20 --amp \
	--opt adam --lr 0.01 --weight-decay 0.01 --no-aug \
	--pin-mem --apex-amp --use-multi-epochs-loader --mean 0.1307 --std 0.3081 --dataset-download --log-interval 100 \
	--alpha 0.25 --num_layers 7 --beta 1 --t_zero 6 --experiment mnist_025_1_6_7_test
