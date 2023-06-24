save_path=work_dirs/distill/cifar10_ConvNetD3_experts200_50e_10ipc_30synsteps_zca
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=6 python eval.py \
    --dataset=CIFAR10 --model=ConvNetD3 --ipc=10 --num_eval 5\
    --syn_steps=30 --expert_epochs=2 \
    --max_start_epoch=15 --zca --lr_img=100 \
    --lr_lr=1e-05 --lr_teacher=0.03 \
    --buffer_path=work_dirs/cifar10_ConvNetD3_experts200_50e \
    --syn_image logged_files/CIFAR10/giddy-frog-2/images_5000.pt \
    --syn_label logged_files/CIFAR10/giddy-frog-2/labels_5000.pt \
    --data_path=../dataSet/cifar10 2>&1 | tee $save_path/$now.txt