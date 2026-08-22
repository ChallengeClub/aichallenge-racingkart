# PilotNet用SW教師データ収集

AI部門の推論コードと混ぜずに、SW部門のMPCを教師として速度別rosbagを収集するための作業領域です。

## ディレクトリ構成

```text
data_collection/
├── collect_sw_teacher.bash       # AWSIM + MPC + rosbag収集
├── extract_collection.bash       # PilotNet用NumPy配列へ変換
├── analyze_collection.py         # 速度・停止率・トピック数の品質集計
├── prepare_sequence_split.py     # 走行単位のtrain/val分割
├── evaluate_pilotnet.py          # PilotNet単独の複数回AWSIM評価
└── collections/                  # 生成物。Git管理外
    └── <collection_id>/
        ├── collection.yaml       # 収集条件
        ├── raw/<sequence>/       # rosbag (MCAP)
        ├── extracted/<sequence>/ # images/steers/accelerations.npy
        ├── train/<sequence>/     # train用view（画像はsymlink）
        └── val/<sequence>/       # validation用view（画像はsymlink）
```

rosbag、画像配列、ログは大容量かつ環境情報を含み得るため、`collections/` 以下は `.gitignore` で除外しています。外部公開や共有の前に必ず内容と利用条件を確認してください。

## 収集

リポジトリトップから実行します。既定では15、20、25 km/hを各150秒収集します。

```bash
aichallenge/ml_workspace/pilot_net/data_collection/collect_sw_teacher.bash
```

速度・収録時間・コレクション名は環境変数で変更できます。

```bash
COLLECTION_ID=sw_mpc_citycircuit_01 \
SPEEDS_KMH="15 20 25" \
RECORD_SECONDS=150 \
aichallenge/ml_workspace/pilot_net/data_collection/collect_sw_teacher.bash
```

同じコレクションへ検証用の2走行目を追加する例です。

```bash
COLLECTION_ID=sw_mpc_citycircuit_01 \
APPEND_COLLECTION=1 RUN_ID=02 \
SPEEDS_KMH="15 20 25" RECORD_SECONDS=150 \
aichallenge/ml_workspace/pilot_net/data_collection/collect_sw_teacher.bash
```

収集中だけ`CONTROL_METHOD=mpc`をコンテナへ渡します。通常起動では未指定のため、AI部門用の既定値`rl_train`のままです。MPCのグローバル速度上限と全区間の参照速度をROSパラメータで変更し、MPC設定ファイル自体は書き換えません。走行ごとにAWSIMとAutowareを新規起動し、古い自己位置・制御状態が混ざることを防ぎます。記録前にWheel Odometryが0.2 m/s以上になったことを確認し、再始動できていない走行は保存しません。

## 抽出と分割

```bash
aichallenge/ml_workspace/pilot_net/data_collection/extract_collection.bash \
  aichallenge/ml_workspace/pilot_net/data_collection/collections/<collection_id>
```

抽出後、走行単位でtrain/validationへ分割します。

```bash
python3 aichallenge/ml_workspace/pilot_net/data_collection/prepare_sequence_split.py \
  --collection aichallenge/ml_workspace/pilot_net/data_collection/collections/<collection_id>
```

同一走行の隣接フレームをtrainとvalidationへ混在させないため、既存のフレーム単位ランダム分割ではなく、シーケンス単位で分割します。
停止率が20%を超える走行は品質レポートに理由を残し、学習用分割から自動除外します。
MPCの加速度はPilotNet出力域を超える場合があるため、`extracted/`には生値を保存し、`train/`と`val/`の学習用viewでは既存前処理と同じ`[-1, 1]`クリップを適用します。

生成したコレクションを直接学習する例です。

```bash
cd <REPO_ROOT>/aichallenge/ml_workspace/pilot_net
python3 train.py \
  data.train_dir=/aichallenge/ml_workspace/pilot_net/data_collection/collections/<collection_id>/train \
  data.val_dir=/aichallenge/ml_workspace/pilot_net/data_collection/collections/<collection_id>/val
```

## PilotNet単独の反復評価

評価用設定では、SACの操舵・加速残差をともに`0.0`にします。AWSIMとAutowareを起動後、Autowareコンテナ内で次のように実行します。

```bash
cd /aichallenge/ml_workspace/reinforcement_learning
ROS_DOMAIN_ID=1 python3 \
  /aichallenge/ml_workspace/pilot_net/data_collection/evaluate_pilotnet.py \
  --config <zero_residual_config.yaml> \
  --episodes 10 \
  --max-steps 1800 \
  --output <evaluation_result.json>
```

エピソードごとの完走有無、終了区間、平均・最高速度、初回周回ステップと、全体の完走率をJSONへ保存します。出力先は`collections/`以下とし、モデルやrosbagと同様にGit管理対象外にします。

## 今回の収集結果

`sw_mpc_speed_profiles_v2`として、citycircuitで15、20、25 km/h設定を各2走行（各約120秒）確保しました。

- 有効サンプル: 6,763画像
- train: 各速度のrun01
- validation: 15/25 km/hのrun03、20 km/hのrun02
- 画像: RGB、66×200、上部37.5%クロップ済み
- 画像―制御同期誤差: 平均約5.6～6.6 ms、最大約51 ms
- 有効走行の停止率: 0%
- 停止していた15/25 km/hのrun02は品質ゲートで除外

実速度の平均は、15 km/h設定で約3.8～3.9 m/s、20 km/h設定で約4.5 m/s、25 km/h設定で約4.6～4.7 m/sでした。20→25 km/hの差が小さいのは、コース形状とMPCの横加速度・加速度制約が支配的になるためです。

## 注意

- SW教師はGNSS・IMU・走行ラインを使用しますが、生成したPilotNetの入力はカメラ画像です。
- 教師生成時の特権情報利用が大会規則上認められるかは、提出前に運営へ確認してください。
- 収集後は完走可否、画像欠損、制御値範囲、画像と制御の同期誤差を確認してから学習へ使います。
- 他チームのrosbagをこの領域へ入れる場合、再学習への利用許諾を別途確認してください。
