DIR_NAME="cifar_base"
PORT=13425

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port $PORT  main_pretrain.py \
--output_dir $DIR_NAME \
--log_dir $DIR_NAME \
--batch_size 64 \
--model mae_vit_base_patch32 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 500 \
--save_epochs 500 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar 

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port $PORT  main_finetune.py \
    --output_dir finetune_$DIR_NAME \
    --log_dir finetune_$DIR_NAME \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch32 \
    --finetune $DIR_NAME/checkpoint-499.pth \
    --epochs 100 \
    --save_epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar