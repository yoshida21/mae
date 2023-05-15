(参考文献)
https://github.com/facebookresearch/mae/issues/120
https://github.com/facebookresearch/mae/blob/main/FINETUNE.md

--master_port 13940 : ポートを指定して同じマシン上で2つのプロセスを回せる

--rot_pred 

--rot_img - --rot_img_tau 0.1　；　クラストークンから予測、ターゲットは回転画像
--rot_patch --rot_patch_tau 0.1　： 各パッチトークンから予測、ターゲットは元画像




→ 結果確認後、accum を外して再実験、epoch50でも可
（finetuneのみ）
base, rot_img_aug, pred, rot_patch_aug, pred, 
img_pred_-2, 0, patch_pred -2 0
100_base

(要学習、shファイルの改変も必須)
その他100



→ NTTスライド作成・提出、(魅力)
→ NTTドコモES考える
→ サイバーエージェント
 


→ independentに学習を行う