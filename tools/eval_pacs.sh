subset=$1


CUDA_VISIBLE_DEVICES=1 python eval_joint.py \
    --dataset=PACS --subset $subset --model=ResNet18_ImageNet --ipc=10 --num_eval 5\
    --syn_steps=5 --expert_epochs=3 \
    --max_start_epoch=35 --lr_img=100 \
    --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=work_dirs/PACS_ResNet18_experts50_50e/$subset \
    --data_path=../dataSet/PACS --syn_path $2
