# MNIST DDPMの設計

`scratch/DDPM/design.md`（拡散モデル）・`scratch/VAE/design.md`（VAE）・`scratch/LatentDDPM/design.md`（Latent Diffusion）の枠組みをそのまま流用し、データソースをswiss rollからMNISTに、条件付けを回転角度（`angle`、連続値）から数字ラベル（0〜9、離散値）に置き換えた実験。潜在拡散（`kind="latent_dit"`）に加え、VAEを経由せずpixel空間で直接拡散する系統（`kind="pixel_transformer"`、28x28）と、VAE単体の学習（`kind="vae"`）も同じディレクトリにある。

## 全体の流れ

1. MNISTデータセット（28×28グレースケール）を32×32に`interpolate`で引き伸ばして使う。
2. `scratch/VAE/model.py`の`VAE`をMNISTデータで学習する（既存のswiss roll用VAEは使えないため、新しいcheckpointを作る）。
3. 学習済みMNIST用VAEを凍結し、`scratch/LatentDDPM`と同じ要領でlatent空間に落としたデータに対してDiTを学習する。DiTへの条件付けは`angle`引数に数字ラベル（0.0〜9.0のfloat）をそのまま渡すことで実現する。`dit_variant="model_dit"`は`scratch/DDPM`のモデルコードをそのまま使うが、`dit_variant="local_dit"`は`scratch/MNIST/model_dit.py`のコピー（ラベルを`nn.Embedding(10, hidden)`で埋め込む）を使う。

## 既存コードとの関係（新規に作るのはグルーコードとMNIST用データ読み込みのみ）

- `scratch.VAE.model.VAE`（Encoder/Decoder）はデータの中身を問わない設計のため、そのまま再利用する。学習に使うデータだけがswiss rollからMNISTに変わる。
- `scratch.DDPM.data.RandomPickedData`/`InputData`・`scratch.DDPM.noise_scheduler.NoiseScheduler`・`scratch.DDPM.model.build_model`・`scratch.VAE.model.Reparameterize`は`scratch/LatentDDPM`と同じ形でそのまま再利用する。`OneStep`/`Loss`は、`kind="latent_dit"`では`loss_type`（mse/l1）を切り替えるため`scratch/MNIST/train.py`で独自に定義しており、`kind="pixel_transformer"`では`scratch.DDPM.train.OneStep`を使う。
- 新規に書くのは、MNISTデータを読み込む関数と、それを呼び出すTrainSchedule（VAE用・latent DiT用・pixel Transformer用）・main.py・infer系のグルーコードと、`model_dit.py`（local_dit）。
- `sinusoidal_embedding(x, dim)`は内部で`x.float()`しているため、int/floatどちらのTensorを渡しても機構上は問題なく動く（`scratch/DDPM/model/embeddings.py`参照）。

## 構成要素

### function

- fn: BuildMNISTBatch(batch, c, h, w) -> (x0: Tensor(batch,c,h,w), label: Tensor(batch,))
    - `torchvision.datasets.MNIST`から`batch`枚をランダムにサンプリングする（初回呼び出し時にデータセットをダウンロード・キャッシュし、以降は再利用する）
    - 各画像を28×28から`(h,w)`へ`F.interpolate(mode="bilinear")`で引き伸ばす
    - 値域を`[0,1]`から`[-1,1]`に変換する（swiss rollの値域`[-1,1]`に合わせる）
    - `c=1`（MNISTはグレースケール）を前提とする
    - `label`はfloat型のTensor（0.0〜9.0）で返す

- fn: BuildLatentMNISTBatch(vae, batch, c_img, h, w) -> (z, label)
    - `scratch/LatentDDPM/train.py`の`BuildLatentBatch`と全く同じ構造。`BuildSwissRoll`の代わりに`BuildMNISTBatch`を呼ぶ点だけが違う
    - `torch.no_grad()`内で: `x0, label = BuildMNISTBatch(batch, c_img, 4*h, 4*w)` → `encoder_output = vae.encoder(x0)` → `z = Reparameterize(encoder_output.mean, encoder_output.logvar)` → `(z, label)`を返す

- fn: TrainSchedule(net, vae, noise_scheduler, cfg, save_path, load_path, device)
    - `scratch/LatentDDPM/train.py`の`TrainSchedule`と全く同じ構造。`BuildLatentBatch`の代わりに`BuildLatentMNISTBatch`を呼ぶ点だけが違う
    - ループ内: `z, label = BuildLatentMNISTBatch(vae, batch, c_img, h, w)` → `random_picked_data = RandomPickedData(z, noise_scheduler.num_timesteps, angle=label)`（`label`を`angle`引数にそのまま渡す）→ `loss = OneStep(net, optimizer, noise_scheduler, random_picked_data, cfg.latent_dit.loss_type)`（`scratch/MNIST/train.py`の`OneStep`。`scratch.DDPM.train`のMSE固定版ではない）

- fn: VaeTrainSchedule(net, cfg, save_path, load_path, device)
    - `scratch/VAE/train.py`の`TrainSchedule`と全く同じ構造。`BuildSwissRoll`の代わりに`BuildMNISTBatch`を呼ぶ点だけが違う
    - MNIST用VAEを学習するために使う（`scratch.VAE.model.VAE`・`scratch.VAE.train.OneStep`（内部で`Loss`/`ReconLoss`(L1)/`KLLoss`を使う）をそのまま再利用する）

### main.py（train_webui対応後: 1本に統一）

train_webui対応にあたり、`vae_main.py`/`transformer_main.py`/`main.py`の3つに分かれていたエントリーポイントは`main.py`1本（`--mode {train,infer}`）に統合した（train_webui skillの規約により、1ディレクトリに検出されるエントリーポイントは1つに固定するため）。どの実験を動かすかは`config.toml`の`[experiment] kind`（`"latent_dit"` / `"pixel_transformer"` / `"vae"`）で切り替える。

```
venv/bin/python -m scratch.MNIST.main --mode train
venv/bin/python -m scratch.MNIST.main --mode infer  [--ckpt path]
```

- `kind = "vae"`: `BuildMNISTBatch`を使って`vae_train.VaeTrainSchedule`でVAEを学習し、`checkpoints/model_vae_mnist.pt`に保存する。
- `kind = "latent_dit"`: `kind="vae"`で学習したVAE checkpoint（`[vae] load_path`）をロードして凍結し、DiT（`train.build_net`、`[latent_dit] dit_variant`で`local_dit`/`model_dit`を切り替え、`loss_type`でmse/l1を切り替え）をlabel条件付きで`train.TrainSchedule`で学習する。
- `kind = "pixel_transformer"`: VAEを経由せず28x28のpixel空間で直接`transformer_train.TrainSchedule`で学習する（`build_model(cfg.pixel_transformer.model_type, ...)`、現状`transformer`）。

### infer.py / transformer_infer.py / vae_infer.py

各`run(load_path, cfg, device)`が対応する`kind`の推論・可視化を行う（`main.py`の`run_infer`が`kind`で呼び分ける）。`kind="latent_dit"`（`infer.py`）は`scratch/LatentDDPM/infer.py`と同じ要領で、逆拡散のサンプリングをVAE Decoderでpixel化して確認する。`label`は0〜9の整数を明示的に指定して生成する（swiss rollの`angle`のように連続的に等間隔で選ぶのではなく、「0を生成して」「7を生成して」のように狙ったラベルを指定できるようにする）。

## 未決事項

- VAEのlatent解像度(h,w)・base channel等の具体的なハイパーパラメータ
- MNIST datasetのダウンロード先・キャッシュ方法の詳細（torchvisionのデフォルトの挙動に従うか等）
- `KL_WEIGHT`・学習率等の具体的な値

## 決定事項

- 配置場所は`scratch/MNIST/`（新規ディレクトリ）。`scratch/DDPM`・`scratch/VAE`・`scratch/LatentDDPM`本体は変更しない
- 主系統はVAEのlatent空間を経由する（MNIST用に新しくVAEを学習する）。比較用にpixel空間で直接拡散する`kind="pixel_transformer"`も別系統として置いている
- 28×28→32×32のリサイズは`interpolate`で引き伸ばす（パディングはしない）
- latent拡散のモデルは`dit_variant`で`local_dit`（`scratch/MNIST/model_dit.py`）か`model_dit`（`build_model("dit", ...)`）を選ぶ
- 条件付けは、`angle`引数に数字ラベル（0.0〜9.0のfloat）をそのまま渡すことで実現する（`local_dit`はこれをlongにして`nn.Embedding`で引く。それ以外のモデルは`scratch/DDPM`のコードを変更せずsinusoidal embeddingで扱う）
- MNISTデータの読み込みには`torchvision.datasets.MNIST`を使う（venvにインストール済み）
