save_path=work_dirs/PACS_ResNet18_experts50_50e
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=6 python buffer.py \
    --batch_train 64 --batch_real 64 \
    --dataset=PACS --model=ResNet18_ImageNet \
    --train_epochs=50 --num_experts=50 \
    --buffer_path=work_dirs/PACS_ResNet18_experts50_50e \
    --data_path=../dataSet/PACS --subset cartoon 2>&1 | tee $save_path/$now.txt