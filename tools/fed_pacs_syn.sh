subset=$1
save_path=work_dirs/fedavg/synthetic_PACS_ResNet18_ImageNet_$subset
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=5 python fed_oracle_syn.py \
    --dataset=PACS --subset $subset --model=ResNet18_ImageNet --ipc=10 --num_eval 1\
    --syn_steps=5 --expert_epochs=3 \
    --max_start_epoch=35 --lr_img=100 \
    --lr_lr=1e-05 --lr_teacher=0.01 --save_dir $save_path  \
    --buffer_path=work_dirs/PACS_ResNet18_experts50_50e/$subset \
    --syn_path $2 --data_path=../dataSet/PACS 2>&1 | tee $save_path/$now.txt