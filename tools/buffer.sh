save_path=work_dirs/cifar10_ConvNetD3_experts200_50e
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=4,5,6,7 python buffer.py \
    --dataset=CIFAR10 --model=ConvNetD3 \
    --train_epochs=50 --num_experts=200 \
    --zca --buffer_path=work_dirs/cifar10_ConvNetD3_experts200_50e \
    --data_path=../dataSet/cifar10 2>&1 | tee $save_path/$now.txt