# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 rot_pred=False, 
                 rot_img=False, rot_img_tau=0, # rot_img_independent=False
                 rot_patch=False, rot_patch_tau=0): # rot_patch_independent=False):
        
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.rot_img = rot_img
        self.rot_patch = rot_patch
        self.rot_pred = rot_pred


        if self.rot_pred:
            if self.rot_img:
                self.rot_img_tau = rot_img_tau
                self.pred_rot_img_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim, bias=True), 
                    nn.ReLU(), nn.Linear(embed_dim, 4, bias=True)
                )
            if self.rot_patch:
                self.rot_patch_tau = rot_patch_tau
                self.pred_rot_patch_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim, bias=True), 
                    nn.ReLU(), nn.Linear(embed_dim, 4, bias=True)
                )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_keep

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, origin_imgs, mask_ratio=0.75):
        transformed_imgs = None

        if self.rot_img:
            transformed_imgs, rot_img_gt = self.rotate_img(origin_imgs) 
        if self.rot_patch:
            transformed_imgs, rot_patch_gt = self.rotate_patch(origin_imgs)  

        imgs = transformed_imgs if transformed_imgs != None else origin_imgs
        latent, mask, ids_restore, ids_keep = self.forward_encoder(imgs, mask_ratio)
        
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        
        target_imgs = transformed_imgs if self.rot_img else origin_imgs 

        rec_loss = self.forward_loss(target_imgs, pred, mask) 

        rot_patch_loss, rot_patch_acc, rot_img_loss, rot_img_acc = \
            [torch.tensor(0.,device=imgs.device) for _ in range(4)]
        if self.rot_pred:
            if self.rot_img:
                rot_img_loss, rot_img_acc = self.predict_img_rot(latent, rot_img_gt)
            if self.rot_patch:
                rot_patch_loss, rot_patch_acc = self.predict_patch_rot(latent, ids_keep, rot_patch_gt)

        return rec_loss, rot_img_loss, rot_img_acc, rot_patch_loss, rot_patch_acc, pred, mask
    
    def rotate_img(self, origin_imgs):
        imgs = copy.deepcopy(origin_imgs) # imgs : b c H W -> b c h p w p
        
        angle = torch.randint(4, (imgs.shape[0],), device=imgs.device)
        for i in range(4):
            imgs[angle==i] = torch.rot90(imgs[angle==i], i, [-2,-1])

        # x = torch.stack([imgs[i] for i in range(4)])
        # x = x * 0.5 + 0.5 #unnorm
        # for i, img in enumerate(x):
        #     img = img.cpu().numpy()
        #     plt.imshow(np.transpose(img,(1,2,0)))
        #     plt.savefig("{}_allrotate.png".format(i))
        # x = torch.stack([origin_imgs[i] for i in range(4)])
        # x = x * 0.5 + 0.5 #unnorm
        # for i, img in enumerate(x):
        #     img = img.cpu().numpy()
        #     plt.imshow(np.transpose(img,(1,2,0)))
        #     plt.savefig("{}.png".format(i))
        # print(angle[0:4])
        # print(test)

        return imgs, angle
    
    def predict_img_rot(self, latent, rot_gt):
        x = latent[:, 0, :] # get cls token, b 1+l*r ed -> b ed
        N, D = x.shape  # batch, dim

        pred_rot = self.pred_rot_img_head(x) # b ed -> b 4

        # rot_gt : b
        accuracy = (torch.argmax(pred_rot, dim=1) == rot_gt).sum() / rot_gt.shape[0]
        accuracy = accuracy.item()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(pred_rot, rot_gt) * self.rot_img_tau # 重みの調整
        return loss, accuracy


    def rotate_patch(self, origin_imgs):
        imgs = copy.deepcopy(origin_imgs)
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p)) # x : b c H W -> b c h p w p
        x = torch.einsum('nchpwq->nhwcpq', x) # x : b c h p w p -> b h w c p p
        
        angle = torch.randint(4, (imgs.shape[0], h, w), device=imgs.device)
        # angle = torch.randint(4, (imgs.shape[0], 1, 1), device=imgs.device).repeat(1, h, w) # same direction

        for i in range(4):
            x[angle==i] = torch.rot90(x[angle==i],i,[-2,-1])

        x = torch.einsum('nhwcpq->nchpwq', x) # x : b h w c p p -> b c h p w p
        x = x.reshape(shape=(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])) # x : b c h p w p -> b c H W
      
        # x = torch.stack([x[i] for i in range(4)])
        # x = x * 0.5 + 0.5 #unnorm
        # for i, img in enumerate(x):
        #     img = img.cpu().numpy()
        #     plt.imshow(np.transpose(img,(1,2,0)))
        #     plt.savefig("{}_rotate.png".format(i))
        # x = torch.stack([origin_imgs[i] for i in range(4)])
        # x = x * 0.5 + 0.5 #unnorm
        # for i, img in enumerate(x):
        #     img = img.cpu().numpy()
        #     plt.imshow(np.transpose(img,(1,2,0)))
        #     plt.savefig("{}.png".format(i))
        # print(angle[0:4])
        # print(test)

        return x, angle
    
    def predict_patch_rot(self, latent, ids_keep, rot_gt):
        # 各tokenから角度を予測
        x = latent[:, 1:, :] # remove cls token, b 1+l*r ed -> b l*r ed
        N, L, D = x.shape  # batch, length, dim
        x = x.reshape(-1, D) # b l*r ed -> b*l*r ed

        pred_rot = self.pred_rot_patch_head(x) # b*l*r ed -> b*l*r 4 

        rot_gt = rot_gt.flatten(1) # rot_gt : b h w -> b h*w(l)
        target = torch.gather(rot_gt, dim=1, index=ids_keep) # b l -> b l*r 
        target = target.reshape(-1) # b l*r -> b*l*r

        accuracy = (torch.argmax(pred_rot, dim=1) == target).sum() / target.shape[0]
        accuracy = accuracy.item()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(pred_rot, target) * self.rot_patch_tau # 重みの調整
        return loss, accuracy


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch32_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch32 = mae_vit_base_patch32_dec512d8b  # decoder: 512 dim, 8 blocks

mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
