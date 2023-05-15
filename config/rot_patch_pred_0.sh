DIR_NAME="cifar_rot_patch_pred_0"
PORT=13440

# OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port $PORT main_pretrain.py \
# --output_dir $DIR_NAME \
# --log_dir $DIR_NAME \
# --batch_size 64 \
# --model mae_vit_base_patch16 \
# --norm_pix_loss \
# --mask_ratio 0.75 \
# --epochs 600 \
# --save_epochs 600 \
# --warmup_epochs 0 \
# --blr 1.5e-4 --weight_decay 0.05 \
# --dataset cifar \
# --rot_pred --rot_patch --rot_patch_tau 1

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port $PORT  main_finetune.py \
    --output_dir finetune_$DIR_NAME \
    --log_dir finetune_$DIR_NAME \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune $DIR_NAME/checkpoint-599.pth \
    --epochs 50 \
    --save_epochs 50 \
    --dist_eval --dataset cifar --nb_classes 10
