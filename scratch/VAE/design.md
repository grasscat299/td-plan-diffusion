# VAEの設計（圧縮・展開部分）

`scratch/DDPM/design.md`と対になるスクラッチ実験。画像 $x$ を圧縮した潜在表現 $z$ に変換し（Encoder）、そこから画像を復元する（Decoder）VAEの設計。外側の構造（クラス・関数のシグネチャとデータの流れ）に加えて、Encoder/Decoderの内部アーキテクチャ（Model節）も決めている。

以降、$h,w$ は**圧縮後（潜在表現）の空間サイズ**を指す。入力画像の空間サイズはその4倍 $(4h,4w)$。

## 構成要素

### class

- Class: EncoderOutput @dataclass
    - elm: mean(shape(batch,c,h,w)), logvar(shape(batch,c,h,w))
    - Encoderの生の出力(batch, c*2, h, w)をchannel方向にc,cで分割し、meanとlogvarとして保持する
    - varianceそのものではなくlog(variance)を保持する（内部的にexp()して使う。負値制約が不要で数値的に安定するため）

- Class: VAEOutput @dataclass
    - elm: x_hat(shape(batch,c,4h,4w)), mean(shape(batch,c,h,w)), logvar(shape(batch,c,h,w))
    - VAE.forwardの戻り値。損失計算に必要な3つをまとめて持つ

- Class: Encoder
    - nn.Moduleを継承
    - method: __init__(c, h, w)
    - method: forward(x: Tensor(batch,c,4h,4w)) -> EncoderOutput
    - 内部アーキテクチャは下のModel節の通り

- Class: Decoder
    - nn.Moduleを継承
    - method: __init__(c, h, w)
    - method: forward(z: Tensor(batch,c,h,w)) -> Tensor(x_hat, shape(batch,c,4h,4w))
    - 内部アーキテクチャは下のModel節の通り

- Class: VAE
    - nn.Moduleを継承
    - elm: encoder: Encoder, decoder: Decoder
    - method: __init__(c, h, w)
        - self.encoder = Encoder(c, h, w)
        - self.decoder = Decoder(c, h, w)
    - method: forward(x: Tensor(batch,c,4h,4w)) -> VAEOutput
        - encoder_output = self.encoder(x)  # EncoderOutput(mean, logvar)
        - z = Reparameterize(encoder_output.mean, encoder_output.logvar)
        - x_hat = self.decoder(z)
        - VAEOutput(x_hat, encoder_output.mean, encoder_output.logvar) を返す

### function

- fn: Reparameterize(mean: Tensor(batch,c,h,w), logvar: Tensor(batch,c,h,w)) -> z: Tensor(batch,c,h,w)
    - std = exp(0.5 * logvar)
    - eps ~ N(0,1)（meanと同じshape）
    - z = mean + std * eps
    - これが「もやのかかった画像（decoderの入力）」を作る処理

- fn: KLLoss(mean: Tensor, logvar: Tensor) -> loss: Tensor
    - 要素ごとに `-0.5 * (1 + logvar - mean^2 - exp(logvar))` を計算
    - 平均を使う（DDPM側のLossと同じ方針）

- fn: ReconLoss(x_hat: Tensor, x: Tensor) -> loss: Tensor
    - L1_lossを使う（要素ごとの|x - x_hat|の平均）

- fn: Loss(output: VAEOutput, x: Tensor, kl_weight: float) -> loss: Tensor
    - ReconLoss(output.x_hat, x) + kl_weight * KLLoss(output.mean, output.logvar)

- fn: OneStep(net, optimizer, x, kl_weight)
    - optimizer.zero_grad()
    - output = net(x)  # VAEOutput
    - loss = Loss(output, x, kl_weight)
    - loss.backward()
    - optimizer.step()

- fn: TrainSchedule(net, cfg, save_path, load_path, device)  # lr, epoch, steps_per_epoch, batch, kl_weight, c, h, w は cfg(config.toml)から取る
    - if load_pathが指定されている
        weightをロードする
    - optimizer = Adam(net.parameters(), lr=lr)
    - for epoch
        - for _ in range(steps_per_epoch)
            - x0, _angle = BuildSwissRoll(batch, c, 4*h, 4*w)  # scratch/DDPM/data.pyを再利用。入力側は4h,4wの解像度。angleはVAEでは使わないため無視する
            - OneStep(net, optimizer, x0, kl_weight)
    - save_pathに保存する
    - （x0_listを事前に一括生成する形だったが、scratch/DDPM/train.pyが既にsteps_per_epochごとに毎回1バッチ生成する方式に変わっているため、それに合わせた）

### Model（内部アーキテクチャ）

SD3などのKL-VAE（Conv+ResBlock、低解像度でのSelf-Attention）を参考に、swiss rollのトイ実験規模に合わせて簡略化した構成にする。

Encoder/Decoderの間にUnetのようなskip connectionは無い（VAEはEncoder/Decoderが完全に独立したネットワークで、zというボトルネックだけを介して繋がる。これがUnetとの一番の違い）。また、t/angleのような条件付けも無いため、ResBlockはDDPM Unetのものよりシンプル（emb injectionが無い）。

#### 約束事

convは全てpaddingありとする（`scratch/DDPM/design.md`のUnetと同じ約束事）。

- `base = c * 16`（`c`は画像のchannel数。DDPM Unetの`_BASE_CHANNEL_MULTIPLIER=32`より小さめにする。t/angleの条件付けMLPが無くパラメータ数が元々少ないため）。
- GroupNormの group数はUnetと同じ考え方で、全ブロック共通で`base`にする（各ブロックのchannel数が必ず`base`の倍数になるようにするため）。
- ResBlockは以下の構造で決める（emb injectionが無い分、Unetの`Resnet`より単純）。channel数が変わる場合は残差側に`conv(in->out)(k=1,s=1)`を挟む（normは通さない）。活性化関数はDDPM Unet（Relu）とは変え、**SiLU**を使う。
```
(b,in,h,w)
->GroupNorm->SiLU->conv(in->out)(k=3,s=1)
(b,out,h,w)
->GroupNorm->SiLU->conv(out->out)(k=3,s=1)
(b,out,h,w)
->+residual（in==outならそのまま加算、in!=outならconv(in->out)(k=1,s=1)を通してから加算。normは通さない）
(b,out,h,w)
```
- downsample（Encoder用）: `conv(ch->ch)(k=3,s=2)`。空間のみ1/2にし、channelは変えない（channel変更は直後のResBlockが担当）。DDPM Unetの`encode_conv`と同じ役割分担。
- upsample（Decoder用）: `nearest補間で2倍`->`conv(ch->ch)(k=3,s=1)`。空間のみ2倍にし、channelは変えない。DDPM Unetの「引き伸ばし」と同じ役割分担。
- AttentionBlock（最小解像度`(h,w)`にのみ1つ挿入）: 1ヘッドのSelf-Attention。**現時点ではコメントアウトして無効化する**（下のEncoder/Decoder図でも無効化した状態で書いている）。将来必要になれば有効化できるよう、構造だけ以下に残しておく。
```
(b,ch,h,w)
->GroupNorm
->flatten+permute
(b,h*w,ch)
->Self-Attention（Q,K,Vはchからの1ヘッドのlinear射影、softmax(QK^T/√ch)Vを計算）
(b,h*w,ch)
->permute+reshape
(b,ch,h,w)
->+residual（入力をそのまま加算。normは通さない）
(b,ch,h,w)
```

#### Encoder

入力側の解像度を`4h,4w`(level1) -> `2h,2w`(level2) -> `h,w`(level3, 最小解像度)の3段階に分け、level間をdownsampleで繋ぐ。

```
(b,c,4h,4w)
->conv_in(c->base)(k=3,s=1)
(b,base,4h,4w)

#level1 (4h,4w)
->ResBlock(base->base)
->ResBlock(base->base)
(b,base,4h,4w)

#joint: downsample
->downsample_conv(base->base)(k=3,s=2)
(b,base,2h,2w)

#level2 (2h,2w)
->ResBlock(base->2*base)
->ResBlock(2*base->2*base)
(b,2*base,2h,2w)

#joint: downsample
->downsample_conv(2*base->2*base)(k=3,s=2)
(b,2*base,h,w)

#level3 (h,w) 最小解像度
->ResBlock(2*base->4*base)
->ResBlock(4*base->4*base)
(b,4*base,h,w)
# ->AttentionBlock(4*base)  # 現時点ではコメントアウトして無効化（shapeは変わらないため以下そのまま続く）
(b,4*base,h,w)
->ResBlock(4*base->4*base)
(b,4*base,h,w)

#出力
->GroupNorm->SiLU
->conv_out(4*base->c*2)(k=3,s=1)
(b,c*2,h,w)
->channel方向にc,cでchunk
mean(b,c,h,w), logvar(b,c,h,w)
```

#### Decoder

Encoderの対称形。level3(h,w, 最小解像度) -> level2(2h,2w) -> level1(4h,4w)の順にupsampleで繋ぐ。skip connectionはEncoder側の値を一切参照しない。

```
z(b,c,h,w)
->conv_in(c->4*base)(k=3,s=1)
(b,4*base,h,w)

#level3 (h,w) 最小解像度
->ResBlock(4*base->4*base)
# ->AttentionBlock(4*base)  # 現時点ではコメントアウトして無効化（shapeは変わらないため以下そのまま続く）
(b,4*base,h,w)
->ResBlock(4*base->4*base)
(b,4*base,h,w)

#joint: upsample
->nearest補間で2倍
->conv(4*base->4*base)(k=3,s=1)
(b,4*base,2h,2w)

#level2 (2h,2w)
->ResBlock(4*base->2*base)
->ResBlock(2*base->2*base)
(b,2*base,2h,2w)

#joint: upsample
->nearest補間で2倍
->conv(2*base->2*base)(k=3,s=1)
(b,2*base,4h,4w)

#level1 (4h,4w)
->ResBlock(2*base->base)
->ResBlock(base->base)
(b,base,4h,4w)

#出力
->GroupNorm->SiLU
->conv_out(base->c)(k=3,s=1)
(b,c,4h,4w)
= x_hat
```

## 未決事項

- `kl_weight`の具体的な値（現在は`scratch/VAE/config.toml`で`1e-3`）
- 上記アーキテクチャ（Attentionは無効化した状態）は実装済みで、WebUIからの学習・推論の実行履歴もある。再構成の品質は未評価
- AttentionBlockを再度有効化するかどうか。無効化したままで再構成品質が足りない場合に検討する

## 決定事項

- `BuildSwissRoll`は`scratch/DDPM/data.py`のものを再利用する（`scratch/VAE/data.py`は作らない）。angleはVAEでは使わないため無視する。
- Encoder/Decoderの内部アーキテクチャはKL-VAE系（Conv+ResBlock）に決定。`base = c * 16`、活性化関数はSiLU（DDPM UnetのReluとは変える）。
- 最小解像度のAttentionBlockは構造だけ残し、コメントアウトして無効化する（swiss rollのトイ規模には過剰と判断）。
