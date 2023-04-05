# 環境構築
python3 -m venv mae_env
source mae_env/bin/activate
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

pip3 install timm==0.3.2
pip3 install tensorboard==2.12.0
// pip3 install submitit==1.4.5
pip3 install matplotlib==3.7.0
pip3 install "numpy<1.24"

https://github.com/facebookresearch/mae/issues/120
https://github.com/facebookresearch/mae/issues/25
nproc_per_node= の後はGPU数
out
--accum_iter  
Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
   
# imagenet pretrain
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir IN_base \
--log_dir IN_base \
--batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 200 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--data_path /srv/datasets/pytorch/ImageNet/

# cifar pretrain
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_base \
--log_dir cifar_base \
--batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar

// finetuneに大きなバグ発見(1000クラスになっていた)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_base \
    --log_dir finetune_cifar_base \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune cifar_base/checkpoint-599.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar 

// linear
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_linprobe.py \
    --output_dir lin_cifar_base \
    --log_dir lin_cifar_base \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune cifar_base/checkpoint-599.pth \
    --epochs 100 \
    --save_epochs 100 \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --dist_eval --dataset cifar --nb_classes 10


// 32 
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_base_32 \
--log_dir cifar_base_32 \
--batch_size 64 \
--model mae_vit_base_patch32 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--save_epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_base_32 \
    --log_dir finetune_cifar_base_32 \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch32 \
    --finetune cifar_base_32/checkpoint-599.pth \
    --epochs 100 \
    --save_epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar

# cifar rot_pred_pretrain
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_rot_pred \
--log_dir cifar_rot_pred \
--batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar \
--rot_aug --rot_pred --rot_head_depth 2 --rot_tau 0.1

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_rot_pred \
    --log_dir finetune_cifar_rot_pred \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune cifar_rot_pred/checkpoint-599.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar

// 32 
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_rot_pred_32 \
--log_dir cifar_rot_pred_32 \
--batch_size 64 \
--model mae_vit_base_patch32 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--save_epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar \
--rot_aug --rot_pred --rot_head_depth 2 --rot_tau 0.1

(aoba)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_rot_pred_32 \
    --log_dir finetune_cifar_rot_pred_32 \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch32 \
    --finetune cifar_rot_pred_32/checkpoint-599.pth \
    --epochs 100 \
    --save_epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar


## -3
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_rot_pred_-3 \
--log_dir cifar_rot_pred_-3 \
--batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar \
--rot_aug True --rot_pred True --rot_head_depth 2 --rot_tau 0.001

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_rot_pred_-3 \
    --log_dir finetune_cifar_rot_pred_-3 \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune cifar_rot_pred_-3/checkpoint-599.pth \
    --epochs 100 \
    --save_epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar

# 3-3
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_rot_pred_3-3 \
--log_dir cifar_rot_pred_3-3 \
--batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--save_epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar \
--rot_aug True --rot_pred True --rot_head_depth 3 --rot_tau 0.001

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_rot_pred_3-3 \
    --log_dir finetune_cifar_rot_pred_3-3 \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune cifar_rot_pred_3-3/checkpoint-599.pth \
    --epochs 100 \
    --save_epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar

# cifar rot_aug_pretrain
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_rot_aug \
--log_dir cifar_rot_aug \
--batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar \
--rot_aug True 

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_rot_aug \
    --log_dir finetune_cifar_rot_aug \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune cifar_rot_aug/checkpoint-599.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar

# img rot
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_rot_img_pred \
--log_dir cifar_rot_img_pred \
--batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--save_epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar \
--rot_aug --rot_all_img --rot_pred --rot_head_depth 2 --rot_tau 0.1

(kou)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_rot_img_pred \
    --log_dir finetune_cifar_rot_img_pred \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune cifar_rot_img_pred/checkpoint-599.pth \
    --epochs 100 \
    --save_epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar

// aug (nene)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--output_dir cifar_rot_img_aug \
--log_dir cifar_rot_img_aug \
--batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 600 \
--save_epochs 600 \
--warmup_epochs 0 \
--blr 1.5e-4 --weight_decay 0.05 \
--dataset cifar \
--rot_aug --rot_all_img --rot_head_depth 2 --rot_tau 0.1

(next)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir finetune_cifar_rot_img_aug \
    --log_dir finetune_cifar_rot_img_aug \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune cifar_rot_img_aug/checkpoint-599.pth \
    --epochs 100 \
    --save_epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --dataset cifar

## Masked Autoencoders: A PyTorch Implementation

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This is a PyTorch/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

* The original implementation was in TensorFlow+TPU. This re-implementation is in PyTorch+GPU.

* This repo is a modification on the [DeiT repo](https://github.com/facebookresearch/deit). Installation and preparation follow that repo.

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### Catalog

- [x] Visualization demo
- [x] Pre-trained checkpoints + fine-tuning code
- [x] Pre-training code

### Visualization demo

Run our interactive visualization demo using [Colab notebook](https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb) (no GPU needed):
<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/147859292-77341c70-2ed8-4703-b153-f505dcb6f2f8.png" width="600">
</p>

### Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in the paper, converted from TF/TPU to PT/GPU:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>8cad7c</tt></td>
<td align="center"><tt>b8b06e</tt></td>
<td align="center"><tt>9bdbb0</tt></td>
</tr>
</tbody></table>

The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).

By fine-tuning these pre-trained models, we rank #1 in these classification tasks (detailed in the paper):
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-B</th>
<th valign="bottom">ViT-L</th>
<th valign="bottom">ViT-H</th>
<th valign="bottom">ViT-H<sub>448</sub></th>
<td valign="bottom" style="color:#C0C0C0">prev best</td>
<!-- TABLE BODY -->
<tr><td align="left">ImageNet-1K (no external data)</td>
<td align="center">83.6</td>
<td align="center">85.9</td>
<td align="center">86.9</td>
<td align="center"><b>87.8</b></td>
<td align="center" style="color:#C0C0C0">87.1</td>
</tr>
<td colspan="5"><font size="1"><em>following are evaluation of the same model weights (fine-tuned in original ImageNet-1K):</em></font></td>
<tr>
</tr>
<tr><td align="left">ImageNet-Corruption (error rate) </td>
<td align="center">51.7</td>
<td align="center">41.8</td>
<td align="center"><b>33.8</b></td>
<td align="center">36.8</td>
<td align="center" style="color:#C0C0C0">42.5</td>
</tr>
<tr><td align="left">ImageNet-Adversarial</td>
<td align="center">35.9</td>
<td align="center">57.1</td>
<td align="center">68.2</td>
<td align="center"><b>76.7</b></td>
<td align="center" style="color:#C0C0C0">35.8</td>
</tr>
<tr><td align="left">ImageNet-Rendition</td>
<td align="center">48.3</td>
<td align="center">59.9</td>
<td align="center">64.4</td>
<td align="center"><b>66.5</b></td>
<td align="center" style="color:#C0C0C0">48.7</td>
</tr>
<tr><td align="left">ImageNet-Sketch</td>
<td align="center">34.5</td>
<td align="center">45.3</td>
<td align="center">49.6</td>
<td align="center"><b>50.9</b></td>
<td align="center" style="color:#C0C0C0">36.0</td>
</tr>
<td colspan="5"><font size="1"><em>following are transfer learning by fine-tuning the pre-trained MAE on the target dataset:</em></font></td>
</tr>
<tr><td align="left">iNaturalists 2017</td>
<td align="center">70.5</td>
<td align="center">75.7</td>
<td align="center">79.3</td>
<td align="center"><b>83.4</b></td>
<td align="center" style="color:#C0C0C0">75.4</td>
</tr>
<tr><td align="left">iNaturalists 2018</td>
<td align="center">75.4</td>
<td align="center">80.1</td>
<td align="center">83.0</td>
<td align="center"><b>86.8</b></td>
<td align="center" style="color:#C0C0C0">81.2</td>
</tr>
<tr><td align="left">iNaturalists 2019</td>
<td align="center">80.5</td>
<td align="center">83.4</td>
<td align="center">85.7</td>
<td align="center"><b>88.3</b></td>
<td align="center" style="color:#C0C0C0">84.1</td>
</tr>
<tr><td align="left">Places205</td>
<td align="center">63.9</td>
<td align="center">65.8</td>
<td align="center">65.9</td>
<td align="center"><b>66.8</b></td>
<td align="center" style="color:#C0C0C0">66.0</td>
</tr>
<tr><td align="left">Places365</td>
<td align="center">57.9</td>
<td align="center">59.4</td>
<td align="center">59.8</td>
<td align="center"><b>60.3</b></td>
<td align="center" style="color:#C0C0C0">58.0</td>
</tr>
</tbody></table>

### Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
