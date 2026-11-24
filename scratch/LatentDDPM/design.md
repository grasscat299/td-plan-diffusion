# Latent DDPMの設計

`scratch/DDPM/design.md`（拡散モデル）と`scratch/VAE/design.md`（VAE）を組み合わせ、生画像空間ではなく学習済みVAEのlatent空間でDDPMを学習させる実験。

## 全体の流れ

1. 学習済みVAE（`scratch/VAE/checkpoints/`のcheckpoint）をロードし、`eval()`にして凍結する（勾配を流さない・重みも更新しない）。
2. 毎ステップ、`scratch.DDPM.data.BuildSwissRoll`でpixel空間の画像`x0_pixel`（shape `(batch,c_img,4h,4w)`）を生成する。
3. 凍結したVAEの`Encoder`で`x0_pixel`を`EncoderOutput(mean,logvar)`に変換し、`scratch.VAE.model.Reparameterize`でサンプリングした`z`（shape `(batch,c_latent,h,w)`）を作る（`torch.no_grad()`内で行い、VAEには勾配を流さない）。
4. `z`を「DDPMのx0」として扱い、`scratch.DDPM.data.RandomPickedData`/`NoiseScheduler`/`Model`/`OneStep`/`Loss`をそのまま再利用してノイズ化→denoise学習する。

## 既存コードとの関係（新規に作るのはグルーコードのみ）

`RandomPickedData`/`InputData`/`NoiseScheduler`/`Model`（`build_model`経由）/`Loss`/`OneStep`は`(batch,C,H,W)`の汎用テンソルとして書かれており、pixel空間かlatent空間かを区別するロジックを持たないため、そのまま流用できる。`scratch.DDPM.train`から`Loss`/`OneStep`を直接importする。`Model`（denoiseネットワーク）は`build_model(name, c=c_latent, h=h, w=w)`のように、画像chの代わりにVAEのlatent ch/h/wを渡すだけで対応する。

## 構成要素

### function

- fn: BuildLatentBatch(vae, batch, c_img, h, w) -> (z: Tensor(batch,c_latent,h,w), angle: Tensor(batch,))
    - `x0_pixel, angle = BuildSwissRoll(batch, c_img, 4*h, 4*w)`
    - `torch.no_grad()`内で:
        - `encoder_output = vae.encoder(x0_pixel)`
        - `z = Reparameterize(encoder_output.mean, encoder_output.logvar)`
    - `(z, angle)` を返す
    - `c_latent`はvaeのEncoderが決める値（VAE構築時の`c`引数と同じ）であり、この関数の引数には現れない

- fn: TrainSchedule(net, vae, noise_scheduler, save_path, load_path, lr, epoch, steps_per_epoch, batch, c_img, h, w, device)
    - if load_pathが指定されている
        - netのweightをロードする（vaeは呼び出し側で既にロード・凍結済みとする）
    - `vae.eval()`（念のためここでも凍結を保証する）
    - `net.to(device)`; `noise_scheduler.to(device)`; `vae.to(device)`
    - `optimizer = Adam(net.parameters(), lr=lr)`（`vae.parameters()`は含めない）
    - for epoch
        - for `_` in `range(steps_per_epoch)`
            - `z, angle = BuildLatentBatch(vae, batch, c_img, h, w)`
            - `random_picked_data = RandomPickedData(z, noise_scheduler.num_timesteps, angle)`
            - `loss = OneStep(net, optimizer, noise_scheduler, random_picked_data)`（`scratch.DDPM.train`のものをそのまま使う）
    - `save_path`に`net`のstate_dictを保存する（`vae`は保存しない。VAE側のcheckpointが別途存在するため）

### main.py（エントリーポイントの想定）

    venv/bin/python -m scratch.LatentDDPM.main --mode train
    venv/bin/python -m scratch.LatentDDPM.main --mode infer  [--ckpt path]

- train_webui対応化に伴い、`VAE_LOAD_PATH`・`VAE_IMAGE_CHANNELS`・`VAE_LATENT_HEIGHT`・`VAE_LATENT_WIDTH`・`MODEL_TYPE`・`NUMBER_OF_DIFFUSION_TIMESTEPS`は`main.py`へのハードコードをやめ、`config.toml`の`[vae]`/`[model]`/`[diffusion]`セクションに移した（`config.toml`はこのプロジェクト専用でWebUIからしか上書きされないため、`scratch.VAE.main`・`scratch.DDPM.main`の値に引きずられないという元の目的は変わらない）
- VAEを`VAE(c=cfg.vae.image_channels, h=cfg.vae.latent_height, w=cfg.vae.latent_width)`で構築し、`cfg.vae.load_path`から重みをロードする
- DDPM側の`Model`は`build_model(cfg.model.type, c=cfg.vae.image_channels, h=cfg.vae.latent_height, w=cfg.vae.latent_width)`のように、VAEと**同じcfg.vae.\***をそのまま渡して構築する（別々に値を書かないことで、VAEのlatentサイズとDDPMが学習するサイズが自動的に一致する。書き間違いによるズレを構造的に防ぐ）
- `TrainSchedule`を呼び出す

## 未決事項

（現時点で無し。VAE checkpoint・DDPMのMODEL_TYPEともに決定済み）

## 決定事項

- 配置場所は`scratch/LatentDDPM/`（新規ディレクトリ）。`scratch/DDPM/`・`scratch/VAE/`と同じ粒度の独立した実験として扱う
- VAEは凍結（`eval()`+`no_grad()`）し、オンザフライでエンコードする（事前キャッシュはしない）。理由: `BuildSwissRoll`はそもそも毎回新しいデータを無限に生成する仕組みなので事前キャッシュする意味が薄く、Latent Diffusionの標準的なやり方（VAEは固定の圧縮器として使う）にも合う
- latentは`Reparameterize`でサンプリングした値を使う（mean決定的にはしない）。VAE自体がサンプリングした値を使って学習しているため、DDPM側もその実際の分布（ノイズを含む）を見て学習する方が一貫性がある
- `RandomPickedData`/`InputData`/`NoiseScheduler`/`Model`/`Loss`/`OneStep`は`scratch.DDPM`のものをそのまま再利用し、`scratch/LatentDDPM/`には`BuildLatentBatch`とVAE対応版`TrainSchedule`、`main.py`のみを新規に書く
- 使用するVAE checkpointは`scratch/VAE/checkpoints/model_vae_128.pt`。この重みに対応する`image_channels`・`latent_height`・`latent_width`は`scratch/LatentDDPM/config.toml`の`[vae]`に書き、`scratch.VAE.main`の値は参照しない（あちらは別実験のたびに書き換わる値のため）
- DDPM側の`Model`は`type="dit"`を使う。`[model] type`・`[diffusion] num_timesteps`も`scratch.DDPM.main`からimportせず`scratch/LatentDDPM/config.toml`に書く（同じ理由。DDPM側の実験のたびに書き換わる値に引きずられないようにする）。`c,h,w`はVAE側と同じ`cfg.vae.*`を使い回すことで自動的にサイズを揃える（上記参照）
