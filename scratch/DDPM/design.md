# 拡散モデルのノイズ化からdenoiseまでの設計

## ノイズ化部分
### class
- Class: RandomPickedData @dataclass
    - elm: x0(画像データ，shape(batch,C,H,W)), t(shape(batch,)), angle(swiss rollの回転角度，shape(batch,))
    - initでx0・num_timesteps(T)・angleを受け取り，tを1~Tまでの一様分布からbatchそれぞれ独立に選ぶ
    - angleはBuildSwissRollが生成した値をそのまま渡す（サンプリングし直さない）
- Class: InputData @dataclass
    - elm: xt(noise化された画像データ，shape(batch,C,H,W)), t(shape(batch,)), eps(noise()で使ったN(0,1)のノイズ，shape(batch,C,H,W)), angle(shape(batch,))
- Class: NoiseScheduler
    - elm: a_t_bar（βtから算出される累積積 ᾱ_t，shape(T,)）
    - method: add_noise(data: RandomPickedData) -> InputData
        - a_t_bar を持っており，tによってnoise化を一発で行いInputDataを作って出力する
- Class: Model
    - nn.Moduleを継承
    - method: __init__(c, h, w)
    - method: forward(data: InputData) -> Tensor(noise_pred, shape(batch,C,H,W))

### function
- fn: BuildSwissRoll(batch, c, h, w) -> (x0, angle)  (x0はshape(batch,C,H,W)、angleはshape(batch,))
    - 2次元swiss roll画像(batch,c,h,w)を1バッチ分生成する（呼ぶたびに新しい画像を生成する。TrainScheduleがstepごとに呼ぶ）
    - x0とあわせて、各画像生成時に使った回転角度angle(shape(batch,))も返す

- fn: Loss(noise_pred: Tensor(modelの出力，shape(batch,C,H,W)), eps: Tensor) -> loss: Tensor
    - 平均を使う
    - MSE_lossを使う

- fn: OneStep(net, optimizer, noise_scheduler, random_picked_data)
    - optimizer.zero_grad()
    - input_data = noise_scheduler.add_noise(random_picked_data)  # εをサンプリングしxtを作る
    - net(input_data) -> noise_pred
    - Loss(noise_pred, input_data.eps) -> loss
    - loss.backward()
    - optimizer.step()

- fn: TrainSchedule(net, noise_scheduler, cfg, save_path, load_path, device)  # lr, epoch, steps_per_epoch(旧num_batches), batch, c, h, w はcfgから読む
    - if load_pathが指定されている
        weightをロードする
    - optimizer = Adam(net.parameters(), lr=lr)
    - for epoch
        - for steps_per_epoch回
            - x0, angle = BuildSwissRoll(batch, c, h, w)  # stepごとに新規生成
            - random_picked_data = RandomPickedData(x0, T, angle)
            - OneStep(net, optimizer, noise_scheduler, random_picked_data)
    - save_pathに保存する

### Model
#### transformer
（コードでの決め打ち値: transformer層数N=4（nn.TransformerEncoderLayer, head数4, FFN次元=c*16*4, post-LN, dropout0.1）。c*16->c*8の間にReLUが入る。2次元positional encodingは固定のsinusoidal。）

InputData.t(batch)
-> sinusoidal embedding
(b,c*16)
-> linear
(b,c*16)
->silu
->linear(c*16->c*16)
(b,c*16)

InputData.angle(batch)
-> sinusoidal embedding
(b,c*16)
-> linear
(b,c*16)
->silu
->linear(c*16->c*16)
(b,c*16)
（t_embと同じ構造・同じ式だが、重みは別（angle専用のlinearを持つ））

InputData(batch,c,h,w)
-> flat
(b,c,h*W)
-> permute
(b,h*W,c)
-> linear
(b, h*W, c*16)

t_emb・angle_embを加算
->(b,h*W,c*16)+(b,*,c*16)+(b,*,c*16)
(b,h*w,c*16)

->2次元positional encoding(row/col)
(b,h*w,c*16)

-> transformer * N
(b, h*W, c*16)
-> linear(c*16->c*8->c)
(b, h*W, c)

#### Unet
##### 約束事
以下の"c"はUnet内部の基準channel数で、コードでは 画像channel数×32（画像1channelなら32）。
convは全てpaddingありとする．
conv(1->16)と書いたら，channelを1から16に拡大することを意味する．

- emb(t), emb(angle)はUnet全体で1回だけ計算し，全てのResnet/encode_Resnet/decode_Resnetブロックに共通で渡す（値は使い回し，ブロックごとに計算し直さない）．

InputData.t(batch)
-> sinusoidal embedding PE(t)
(b,c*4)
-> linear(c*4->c*8) [W1,b1]
(b,c*8)
-> swish
(b,c*8)
-> linear(c*8->c*4) [W2,b2]
(b,c*4) : emb(t)

InputData.angle(batch)についても同じ構造（PE->linear->swish->linear）でemb(angle)(b,c*4)を計算する．重みはemb(t)用とは別（angle専用のW1,b1,W2,b2を持つ）．

- 各Resnet/encode_Resnet/decode_Resnetブロックは，1つ目のconv後・2つ目のGroupNorm+Reluの前で，emb(t)とemb(angle)をそれぞれそのブロックの（1つ目のconv直後の）channel数へlinearで射影してから加算する．
    - 射影用のlinear（linear_t, linear_angle）はブロックごとに個別の重みを持つ（channel数がブロックによって異なるため．t用とangle用でも別重み）．
    - emb(t)・emb(angle)はshape(b,c*4)なので，linear_t/linear_angleでshape(b,channel数)に射影した後，(b,channel数,1,1)として(b,h,w,channel数)にbroadcast加算する．

- 全てのconvの直前にGroupNorm->Reluを挟む（pre-norm）．残差結合の"+"自体にはnormをかけず，入力をそのまま加算する（恒等写像の勾配経路を保つため）．GroupNormのgroup数は，どのブロックのchannel数も基準channel数cの倍数（c, 2c, 3c, 4c, 6c）になっていることを利用し，全ブロック共通でgroup数=cとする（channel数/group数が必ず整数になる）．

- Resnetは以下のように決める．
(b,h,w,c)
->GroupNorm
(b,h,w,c)
->Relu
(b,h,w,c)
->conv(c->c)(k=3,s=1)
(b,h,w,c)
->+linear_t(emb(t))，+linear_angle(emb(angle))（channel数cへ射影してbroadcast加算）
(b,h,w,c)
->GroupNorm
(b,h,w,c)
->Relu
(b,h,w,c)
->conv(c->c)(k=3,s=1)
(b,h,w,c)
->+(b,h,w,c)(残差結合で入力をたす．normを通さない)
(b,h,w,c)

- encode_Resnetは以下のように決める．
(b,h,w,c)
->GroupNorm
(b,h,w,c)
->Relu
(b,h,w,c)
->conv(c->2c)(k=3,s=1)
(b,h,w,2c)
->+linear_t(emb(t))，+linear_angle(emb(angle))（channel数2cへ射影してbroadcast加算）
(b,h,w,2c)
->GroupNorm
(b,h,w,2c)
->Relu
(b,h,w,2c)
->conv(2c->2c)(k=3,s=1)
(b,h,w,2c)

#残差のc->2c（normを通さない）
input(b,h,w,c)
->conv(c->2c)(k=1,s=1)
(b,h,w,2c)

->+(b,h,w,2c)(残差結合で入力をたす)
(b,h,w,2c)

- decode_Resnetは以下のように決める．
(b,h,w,3c)
->GroupNorm
(b,h,w,3c)
->Relu
(b,h,w,3c)
->conv(3c->c)(k=3,s=1)
(b,h,w,c)
->+linear_t(emb(t))，+linear_angle(emb(angle))（channel数cへ射影してbroadcast加算）
(b,h,w,c)
->GroupNorm
(b,h,w,c)
->Relu
(b,h,w,c)
->conv(c->c)(k=3,s=1)
(b,h,w,c)

#残差の3c->c（normを通さない）
input(b,h,w,3c)
->conv(3c->c)(k=1,s=1)
(b,h,w,c)

->+(b,h,w,c)(残差結合で入力をたす)
(b,h,w,c)

- encode_convは以下のようにきめる．
(b,h,w,c)
->conv(c->c)(k=3,s=2)
(b,h/2,w/2,c)

- 引き伸ばしは以下のように決める
(b,h,w,c)
->引き伸ばし
(b,2h,2w,c)

- 変な言い方かもしれないが，layerとlayerを繋ぐ部分をjointと呼んでいる

###### model
InputData.t, InputData.angleから，上記約束事の通りemb(t), emb(angle)(各shape(b,c*4))を最初に1回だけ計算する．
以降の全resnet/encode_resnet/decode_resnetブロックにemb(t), emb(angle)をそのまま渡す．

(b,h,w,c_img)
->conv(c_img->c)(k=3,s=1)
(b,h,w,c)

#layer1
->resnet
(b,h,w,c)
->resnet
(b,h,w,c):skipS1

#joint
->encode_conv
(b,h/2,w/2,c)

#layer2
->encode_resnet
(b,h/2,w/2,2c)
->resnet
(b,h/2,w/2,2c):skipS2

#joint
->encode_conv
(b,h/4,w/4,2c)

#layer3
->encode_resnet
(b,h/4,w/4,4c)
->resnet
(b,h/4,w/4,4c)

#joint
->引き伸ばし
(b,h/2,w/2,4c)

#layer2
->conv(4c->4c)(k=3,s=1)
(b,h/2,w/2,4c)
->concat(skipS2)
(b,h/2,w/2,6c)
->decode_resnet
(b,h/2,w/2,2c)
->resnet
(b,h/2,w/2,2c)

#joint
->引き伸ばし
(b,h,w,2c)

#layer1
->conv(2c->2c)(k=3,s=1)
(b,h,w,2c)
->concat(skipS1)
(b,h,w,3c)
->decode_resnet
(b,h,w,c)
->resnet
(b,h,w,c)

->conv(c->c_img)(k=3,s=1)
(b,h,w,c_img)

#### DiT
（コードでの決め打ち値: 以下の"c*16"・"c*32"はコードではそれぞれ hidden_dim=c*64・c*128（MLP倍率2は一致）。パッチサイズp=2、DiT Block数N=8、head数4。）


InputData.t(b,)
-> sinusoidal embedding
(b,c*16)
-> linear
(b,c*16)
->silu
->linear(c*16->c*16)
(b,c*16) : t_emb

InputData.angle(b,)についても同じ構造（sinusoidal embedding->linear->silu->linear）でangle_emb(b,c*16)を計算する．重みはt_emb用とは別（angle専用のlinearを持つ）．

t_emb + angle_emb
= 条件ベクトル y (b,c*16)

InputData.xt(b,c,h,w)
-> patchify: p×pごとに分割してflatten（コードではp=2。h,wを割り切るサイズをモジュール定数として決め打ち）
(b, (h/p)*(w/p), p*p*c)
-> linear
(b, (h/p)*(w/p), c*16)
-> 2次元sin-cos位置埋め込みを加算（固定・学習しない）
(b, (h/p)*(w/p), c*16)

-> DiT Block（adaLN-Zero） * N
    [ブロック内部]
    y -> linear -> (b, c*16*6) -> 分割 -> γ1,β1,α1,γ2,β2,α2 各(b,c*16)

    x -> LayerNorm -> γ1,β1でscale-shift
      -> Multi-Head Self-Attention
      -> α1でscale
      -> x + (残差接続)
    (b, (h/p)*(w/p), c*16)

    -> LayerNorm -> γ2,β2でscale-shift
      -> linear(c*16->c*32) -> silu -> linear(c*32->c*16)
      -> α2でscale
      -> 残差接続
    (b, (h/p)*(w/p), c*16)

(b, (h/p)*(w/p), c*16)
-> 最終adaLN（γ,βのみ、ゲートαはなし）
-> linear(c*16 -> p*p*c)
(b, (h/p)*(w/p), p*p*c)
-> unpatchify: パッチを空間配置に戻す
(b, c, h, w)
= noise_pred
