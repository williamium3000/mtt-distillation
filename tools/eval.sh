save_path=work_dirs/distill/cifar10_ConvNetD3_experts200_50e_10ipc_30synsteps_zca
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=6 python eval.py \
    --dataset=CIFAR10 --model=ConvNetD3 --ipc=10 --num_eval 5\
    --syn_steps=30 --expert_epochs=2 \
    --max_start_epoch=15 --zca --lr_img=100 \
    --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=work_dirs/cifar10_ConvNetD3_experts200_50e \
    --syn_image 
    --data_path=../dataSet/cifar10 2>&1 | tee $save_path/$now.txt