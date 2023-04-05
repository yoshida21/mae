bash config/base32.sh


--master_port 13940 : ポートを指定して同じマシン上で2つのプロセスを回せる

--rot_pred 

--rot_img - --rot_img_tau 0.1　；　クラストークンから予測、ターゲットは回転画像
--rot_patch --rot_patch_tau 0.1　： 各パッチトークンから予測、ターゲットは元画像

→ independentに学習を行う