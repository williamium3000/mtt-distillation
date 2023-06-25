save_path=work_dirs/PACS_ResNet18_experts50_50e
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=4,5,6,7 python buffer.py \
    --dataset=PACS --model=ResNet18 \
    --train_epochs=50 --num_experts=50 \
    --zca --buffer_path=work_dirs/PACS_ResNet18_experts50_50e \
    --data_path=../dataSet/PACS --subset  2>&1 | tee $save_path/$now.txt