# PolylineInterp 設計

## 概要
checkpoint (x,y) をN個入力すると、それらを順番に繋いだ経路上に等間隔に並ぶ点列を予測するモデル。
本実験ではまず N=3（三角形の3頂点）を toy データとし、checkpoint0→1→2→0 と繋いで**閉じた経路（3辺）**とする。
経路の全長（弧長）はデータごとに異なるため、出力点数 `L_actual` もデータごとに異なる。学習時は
データセット内の最大値 `L_max` 次元で出力し、余った分は0paddingしてlossから除外する。

N=3・閉路以外への拡張（開いた折れ線、N可変など）は今回のスコープ外。将来拡張する場合は
`GenerateCheckpoints`・`PolylineArcLength`・`SampleEquidistantPoints` の `closed` 引数や
`n_checkpoints` を差し替えるだけで対応できるように設計しておく。

## ディレクトリ構成（model-dev-convention準拠）
```
scratch/PolylineInterp/
  main.py    # --modeで学習/推論を切り替える唯一のCLIエントリーポイント
  config.example.toml
  config.py  # config.toml の読み込み・checkpoint同梱configの復元
  sync_config.py
  data.py
  model/
    __init__.py
    model.py
  train.py   # 学習ループ（main.py --mode train から呼ばれる）
  test.py
  infer.py   # 推論ロジック（main.py --mode infer から呼ばれる）
  checkpoint.py
  checkpoints/.gitkeep
```

## config.toml 項目
```
[data]
n_checkpoints = 3      # N。今回は三角形固定
closed = true           # 閉じた経路（頂点を一周）にするか
coord_min = -1.0
coord_max = 1.0
step = 0.1              # 等間隔サンプリングの間隔（座標と同スケール）
min_triangle_area = ... # 退化三角形（面積が小さすぎる）を弾く閾値
num_train_samples = ...
num_val_samples = ...
seed = ...

[model]
d_model = ...
n_heads = ...
n_encoder_layers = ...
n_decoder_layers = ...
dim_feedforward = ...
head_type = "interp"    # "linear" or "interp"

[train]
lr = ...
epoch = ...
batch = ...

[wandb]
project = ...
group = ...
```

`l_max` はデータセットを実際に生成してみないと決まらない値（各サンプルの `L_actual` の最大値）なので
`config.toml` には書かず、`BuildDataset` の戻り値として得てから `Config`（dataclass）にセットする。
checkpoint保存時はこの確定済み `l_max` を含めた `Config` を同梱するので、推論時は `config.toml` を見ずに
モデル構造を復元できる（model-dev-convention の checkpoint 規約通り）。

## データ生成（data.py）

### class
- Class: `Sample` @dataclass
  - elm: `checkpoints`(shape `(N,2)`), `targets`(shape `(L_max,2)`), `mask`(shape `(L_max,)`, bool)

### function
- fn: `GenerateTriangleCheckpoints(coord_min, coord_max, min_triangle_area) -> checkpoints: Tensor(N=3,2)`
  - `[coord_min, coord_max]^2` の一様分布から3点をランダムに生成する
  - 三角形の面積が `min_triangle_area` 未満（ほぼ一直線に並ぶ退化三角形）なら再サンプリングする

- fn: `PolylineArcLength(checkpoints: Tensor(N,2), closed: bool) -> arc_length: float`
  - セグメントを `closed=False` なら `(checkpoints[i], checkpoints[i+1])` for `i in 0..N-2`、
    `closed=True` なら加えて `(checkpoints[N-1], checkpoints[0])` も含める
  - 各セグメントのユークリッド距離の総和を返す

- fn: `SampleEquidistantPoints(checkpoints: Tensor(N,2), step, closed) -> points: Tensor(L_actual,2)`
  - **セグメント（辺）ごとに独立**して等間隔サンプリングする。各セグメントは
    `num_intervals = max(1, round(seg_len/step))` 個の区間に分割し、区間の右端（=区間ごとの終点）を
    順に出力列へ加える。区間の右端は必ずセグメントの終点（=次のcheckpoint）と一致するので、
    **checkpointsは`step`の倍数と一致するかに関わらず必ず出力列に含まれる**
    （2026-09-24: 「stepの都合でcheckpointが出力列から漏れる」問題への対応として、グローバルに
    `0, step, 2*step, ...`と区切る方式から変更した）。
  - 実際の間隔は `seg_len / num_intervals` になり、指定した`step`の近似値になる
    （辺が`step`よりかなり短い場合は`num_intervals=1`、すなわち区間内に中間点は生成されず
    始点→終点の1区間だけになる。この場合も終点＝checkpointは必ず含まれるので衝突は起きない）
  - `L_actual = 1 + sum(num_intervals for 各セグメント)`
  - 1点目は必ず `checkpoints[0]` と一致する

- fn: `BuildDataset(num_samples, n_checkpoints, coord_min, coord_max, step, closed, min_triangle_area) -> (checkpoints_all: Tensor(num_samples,N,2), targets: Tensor(num_samples,L_max,2), mask: Tensor(num_samples,L_max) bool, l_max: int)`
  - `num_samples` 回 `GenerateTriangleCheckpoints` → `SampleEquidistantPoints` を呼び、各サンプルの
    `L_actual` を集める
  - `l_max = max(全サンプルのL_actual)`
  - `targets` を `l_max` 次元に0-paddingし、`mask` は `[0, L_actual)` を `True`、それ以降を `False` にする

## モデル（model/model.py）

### class
- Class: `PolylineTransformer`（`nn.Module`）
  - `__init__(n_checkpoints, l_max, d_model, n_heads, n_encoder_layers, n_decoder_layers, dim_feedforward, head_type)`
    - `head_type`: `"linear"` または `"interp"`（詳細は次項「頂点が丸まる問題への対応」参照）。
      `_HEAD_TYPES = ("linear", "interp")` に無い値なら `ValueError`。
    - `input_proj`: `Linear(2 -> d_model)`
    - `input_pos_emb`: `nn.Parameter(shape(n_checkpoints, d_model))`
      （学習可能。checkpointの並び順（0番目/1番目/2番目）を表す埋め込み。Nは固定なのでsin/cosではなく
      素朴な学習可能パラメータで十分）
    - `encoder`: `nn.TransformerEncoder`（`n_encoder_layers`層, `d_model`, `n_heads`, `dim_feedforward`,
      `batch_first=True`）
    - `query_emb`: `nn.Parameter(shape(l_max, d_model))`
      （decoder入力の学習可能クエリ。出力L_max個の位置それぞれに専用の学習可能ベクトルを持つ、DETR風）
    - `decoder`: `nn.TransformerDecoder`（同様の層構成）
    - `output_head`:
      - `head_type == "linear"`: `Linear(d_model -> 2)`
      - `head_type == "interp"`: `Linear(d_model -> n_checkpoints)`
  - method: `forward(checkpoints: Tensor(batch,N,2)) -> Tensor(batch,l_max,2)`
    ```
    checkpoints (batch,N,2)
    -> input_proj
    (batch,N,d_model)
    -> + input_pos_emb.unsqueeze(0)
    (batch,N,d_model)
    -> encoder                      # Nは固定長なのでsrc_key_padding_mask不要
    (batch,N,d_model) : memory

    query_emb.unsqueeze(0).expand(batch,-1,-1)
    (batch,l_max,d_model) : q

    q, memory -> decoder            # self-attn(query間) + cross-attn(queryからmemoryへ)
    (batch,l_max,d_model) : h

    # head_type == "linear"
    h -> output_head
    (batch,l_max,2) = pred

    # head_type == "interp"
    h -> output_head
    (batch,l_max,n_checkpoints) : attn_logits
    -> softmax(dim=-1)
    (batch,l_max,n_checkpoints) : weights        # 各行の和が1
    weights @ checkpoints
    (batch,l_max,2) = pred                       # checkpoints座標の加重和（凸結合）
    ```

### 頂点が丸まる問題への対応（`head_type`）
- 学習結果でcheckpoint（三角形の頂点）付近が丸まる現象が確認された。原因は、`head_type="linear"`の
  `Linear(d_model,2)`がdecoder出力から座標を**自由に**回帰するため、「ちょうどcheckpointと一致する」
  ことを構造的に保証しないから。角（経路の進行方向が急に変わる不連続点）の近くでは、内部のわずかな
  推定誤差が2方向の平均として可視化されやすく、直線区間の途中よりも誤差が目立ちやすい。
- 対策として`head_type="interp"`を追加した。出力座標を「decoder出力から予測したN個のcheckpointへの
  attention重み（softmax）と、入力checkpoints座標そのものとの加重和」として明示的に計算する
  （Pointer Networks〔Vinyals et al., 2015〕的な発想）。重みが1つのcheckpointに集中すれば、
  その座標を構造的に厳密に再現できるため、`Linear`による近似誤差が入り込む余地がない。
- `config.toml`の`[model].head_type`で切り替える。`config.example.toml`では`"interp"`を既定値とする。

### function
- fn: `MaskedMSELoss(pred: Tensor(batch,l_max,2), target: Tensor(batch,l_max,2), mask: Tensor(batch,l_max) bool) -> loss: Tensor`
  - `diff2 = (pred - target)**2`  # (batch,l_max,2)
  - `mask` を `(batch,l_max,1)` にunsqueezeしてbroadcastし、`diff2 * mask` の総和を `mask.sum() * 2`
    （有効要素数）で割って平均を取る

- fn: `OneStep(net, optimizer, checkpoints, targets, mask)`
  - `optimizer.zero_grad()`
  - `pred = net(checkpoints)`
  - `loss = MaskedMSELoss(pred, targets, mask)`
  - `loss.backward()`
  - `optimizer.step()`

- fn: `TrainSchedule(net, cfg, checkpoints_all, targets, mask, val_checkpoints, val_targets, val_mask, save_path, load_path, device)`（`train.py`。`MaskedMSELoss`・`OneStep` も `train.py` にある）
  - `load_path` が指定されていれば `load_checkpoint` で `state_dict` をロード
  - `optimizer = Adam(net.parameters(), lr=cfg.train.lr)`
  - データは `main.py` の `run_train` が先に train/val を別々に `BuildDataset` で生成し
    （`l_max` は両者の最大値、短い方は0padding）、`cfg.model.l_max` にセットしてモデルを構築してから渡す
  - `for epoch:`
    - `for batch in train_loader:` `OneStep(net, optimizer, batch.checkpoints, batch.targets, batch.mask)`
    - valで `MaskedMSELoss` を計算しログ（train_webui の軸つきグラフ `train/loss`・`epoch/loss`）
  - `save_checkpoint(save_path, net, cfg)`（`l_max` を含む `cfg` を同梱）

### エントリーポイント（main.py, `--mode`切り替え）
- CLIエントリーポイントは`main.py`一本（world_model_v1と同じ流儀）。`argparse`で
  `--mode {train,infer}`（必須）と`--ckpt`（inferのみ、省略可）を受け取る。
  ```
  venv/bin/python -m scratch.PolylineInterp.main --mode train
  venv/bin/python -m scratch.PolylineInterp.main --mode infer  [--ckpt path]
  ```
- `train.py`/`infer.py`はロジック関数（`TrainSchedule`、`infer.run(load_path, device)`）だけを持ち、
  `if __name__ == "__main__":`による単体実行は持たせない。
- `_save_path(cfg) -> str`: `cfg.wandb.group`を含めた名前（`checkpoints/model_{group}.pt`）を組み立てる。
  固定名だと、`head_type`のようにアーキテクチャが違う別実験を同じファイルに上書きしてしまい、
  次回起動時に形の合わない`state_dict`を誤ってロードする事故につながるため
  （model-dev-conventionの「チェックポイント」規約に追記済み）。実験を変える（`head_type`を
  変える等）ときは`config.toml`の`[wandb] group`も変える運用にする。
- `run_train(cfg, device)`: `load_path = _save_path(cfg) if 存在すれば`。`BuildDataset`で
  train/valデータを生成しl_maxを確定 → `net`構築 → `TrainSchedule(..., save_path=_save_path(cfg), ...)`。
- `run_infer(cfg, device, ckpt=None)`: `load_path = ckpt or _save_path(cfg)`。存在しなければ
  `FileNotFoundError`（「先に--mode train / --ckptを確認」）。`infer.run(load_path, device)`を呼ぶ。

## 推論（infer.py）
- `run(load_path: str, device: torch.device) -> None`（`main.py`の`run_infer`から呼ばれる）。
- `load_checkpoint(load_path) -> state_dict, cfg`（`cfg` に `n_checkpoints`, `l_max`, `d_model` 等が
  同梱されているので、手元の `config.toml` に依存せずモデル構造を復元できる）
- `net = PolylineTransformer(**cfgから復元した引数)`; `net.load_state_dict(state_dict)`
- 入力 `checkpoints (N,2)` を `net` に通し `pred (l_max,2)` を得る
- 呼び出し側で `SampleEquidistantPoints(checkpoints, cfg.data.step, cfg.data.closed).shape[0]` から
  実際の `L_actual` を求め、`pred[:L_actual]` を切り出して使う（モデルは常に `l_max` 点を出力するが、
  意味があるのは `L_actual` 点まで）

## 可視化・確認（test.py 等）
- 三角形の3頂点と、モデルの予測点列（`L_actual` まで切り出したもの）をmatplotlibで重ねてプロットし、
  等間隔性・経路への追従を目視確認する（DDPMの `visualize_training_data.py` 相当）
