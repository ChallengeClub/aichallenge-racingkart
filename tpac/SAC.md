# Soft Actor-Critic（SAC）で安定周回を目指す開発メモ

## 1. この文書の目的

この文書は、自動運転AIチャレンジ2026のAWSIM環境で、Soft Actor-Critic（SAC）を使ってレーシングカートを安定して周回させるために行った対応を、強化学習の初学者にも分かるようにまとめたものです。

今回の開発では、SACにすべての運転を最初から学ばせるのではなく、走行ラインを追従する基本制御にSACの補正を加える「残差強化学習」を採用しました。これにより、学習初期からある程度安全に走行でき、SACは基本制御の弱い部分だけを改善できます。

この文書は公式資料ではありません。環境構築方法や競技仕様は、必ず大会の公式ドキュメントと公式リポジトリで最新情報を確認してください。

- 公式SAC解説: <https://automotiveaichallenge.github.io/aichallenge-documentation-racingkart/ml_sample/soft_actor_critic.html>
- 公式ドキュメント: <https://automotiveaichallenge.github.io/aichallenge-documentation-racingkart/>
- 公式リポジトリ: <https://github.com/AutomotiveAIChallenge/aichallenge-racingkart>

## 2. SACとは

SACは、連続値の操作を学習するための強化学習アルゴリズムです。

今回の運転では、エージェントが次の2つの連続値を出力します。

1. 操舵の補正量
2. 加速の補正量

強化学習では、エージェントが環境を観測し、行動を選び、その結果として報酬を受け取ります。このサイクルを繰り返すことで、長期的に大きな報酬を得られる行動を学習します。

```text
AWSIMの状態を観測
    ↓
SACが行動を出力
    ↓
車両へ操舵・加速指令を送信
    ↓
AWSIMが次の状態を返す
    ↓
報酬と終了条件を計算
    ↓
SACのニューラルネットワークを更新
```

SACは探索を重視する仕組みを持っており、さまざまな操作を試しながら学習します。一方、自由な探索をそのまま車両操作へ反映すると、学習初期に壁へ衝突しやすいという問題があります。

## 3. 最初に確認した公式サンプルの状態

公式サンプルは、主に次の構成でした。

- 観測: 64×64ピクセルの前方カメラ画像と車速
- 行動: 操舵と加速
- 報酬: 速度報酬、時間罰則、衝突罰則
- 終了条件: 急減速または低速状態の継続
- 学習アルゴリズム: Stable-Baselines3のSAC

このまま学習を始める前に、公式手順に従って次の設定が必要でした。

### GPU描画の有効化

`.env` の `COMPOSE_FILE` にGPU用Composeファイルを含めます。

```text
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
```

### AWSIMカメラの有効化

`aichallenge/simulator_scripts/dev.sh` でカメラを有効にし、エピソード開始時の待ち時間をなくしました。

```text
--camera cpu
--start-count-seconds 0
```

カメラが無効なままだと、SACへ渡される画像がゼロ埋めになり、画像からコースを学習できません。

### 制御方式の変更

`reference.launch.xml` の既定制御方式を `rl_train` に変更しました。

```xml
<arg name="control_method" default="rl_train"
```

## 4. 最初の方式で分かった問題

最初は、公式サンプルに近い「画像と速度からSACが操舵・加速を直接決める方式」を試しました。

短時間の学習では、次の問題が見つかりました。

### 4.1 速度だけでは正しい方向が分からない

速度報酬が中心だと、エージェントは「速く進むこと」は学べても、「コース中央を走ること」や「次のカーブに沿って曲がること」を直接は評価されません。

その結果、直進して壁へ向かう、低速で停止する、といった方策になることがありました。

### 4.2 カメラ画像だけの学習には時間がかかる

画像からコース形状を理解し、その上で適切な操舵を学ぶには、多数の試行が必要です。公式設定の30万ステップは、そのための長時間学習を想定しています。

数千ステップ程度の短い実験では、安定周回まで到達しませんでした。

### 4.3 非対称な行動空間

加速の行動範囲が `[0, 1]` だったため、Stable-Baselines3から、行動空間を対称な `[-1, 1]` にすることを推奨する警告が出ました。

そこで、SACが扱う行動空間は `[-1, 1]` にし、アダプター内でAWSIM用の加速値 `[0, 1]` に変換するようにしました。

### 4.4 停止判定が弱い

低速判定のしきい値が小さいと、ほぼ停止している方策が長時間エピソードを継続することがあります。しきい値と継続回数を調整し、明らかに進めないエピソードを適切に終了できるようにしました。

## 5. 追加した観測

カメラと速度に加え、走行ラインCSVから、車体を基準にした複数の目標点を観測へ追加しました。

たとえば、車両から見て走行ライン上の点が次のような位置にあることを数値で表します。

```text
直近の点:     前方 1 m、左 0.2 m
少し先の点:   前方 5 m、左 1.0 m
さらに先の点: 前方 10 m、左 3.0 m
```

これにより、SACは画像だけでなく、次のカーブが左右どちらにあるかを直接判断できます。

走行ライン点は、地図座標から車体座標へ変換して使用します。値が大きくなりすぎないよう、設定した距離で割り、`[-1, 1]` に収めています。

### 自己位置受信前の対策

学習・推論プロセスを起動した直後は、自己位置のROSメッセージがまだ届いていないことがあります。

自己位置がない状態を座標 `(0, 0)` として扱うと、走行ラインまで約99 km離れているという誤判定が発生しました。この大きな誤差が報酬へ入ると、学習が不安定になります。

対策として、次の2段階の保護を追加しました。

1. リセット後、自己位置メッセージを受信するまで一定時間待つ
2. 自己位置が欠けている場合は、走行ライン誤差と進捗報酬を無効にする

## 6. 報酬設計の改善

報酬には、次の要素を追加しました。

### 正の報酬

- 前進して速度が出た
- 走行ライン上のインデックスが前へ進んだ
- セクションを通過した
- ラップが進んだ

### 負の報酬

- 衝突した
- 走行ラインから離れた
- 操舵量が大きすぎる
- 前回から操舵を急に変えた
- 前回から加速を急に変えた
- 時間だけが経過した

概念的には次のような式です。

```text
報酬 =
    速度報酬
  + 走行ライン前進報酬
  + セクション通過報酬
  + ラップ報酬
  - 横ずれ罰則
  - 操舵量罰則
  - 操作変化罰則
  - 衝突罰則
  - 時間罰則
```

速度だけでなく走行ライン上の進捗を評価することで、「コースに沿って前へ進む」行動を学びやすくしています。

## 7. 残差SAC

今回、最終的に採用した方式です。

### 7.1 考え方

走行ラインを追従するPure Pursuit相当の基準操作を計算し、SACはその操作に対する小さな補正だけを出力します。

```text
最終操舵 = 基準操舵 + SACの操舵補正
最終加速 = 基準加速 + SACの加速補正
```

基準操舵は、現在位置から少し先の走行ライン点を見て計算します。基準加速は、現在速度が目標速度へ近づくように計算します。

### 7.2 利点

- 学習前でも最低限の走行ライン追従ができる
- SACが極端な行動を出しても、補正幅を制限できる
- SACはカーブや路面状態に応じた微調整へ集中できる
- 直接制御より短い学習で安定走行を確認しやすい

### 7.3 補正幅の制限

最初はSACの補正幅を大きめに設定しましたが、学習済みモデルの決定論的推論で基準操舵を崩し、792ステップで停止しました。

安定性を優先し、最終的には操舵と加速の補正幅をともに `0.02` としました。

これは、SACに運転の全権を渡すのではなく、安定した基本制御をわずかに補正させる設定です。

## 8. 走行ラインのインデックス処理

使用した走行ラインCSVには、ガレージからコースへ接続する区間と、周回する区間が同じファイルに含まれています。

単純にCSV末尾から先頭へ戻すと、周回中にガレージ区間へ戻ろうとしてしまいます。そこで、周回開始インデックスを設定し、CSV末尾へ到達した後はガレージ先頭ではなく周回開始点へ戻るようにしました。

また、毎ステップ全走行ライン点から最近傍点を選ぶと、近接する別区間へインデックスが飛ぶことがあります。前回の最近傍点の周辺だけを探索することで、走行ライン上の進行が連続するようにしました。

## 9. 学習設定

最終実験では、主に次の設定を使用しました。

```yaml
algorithm:
  name: sac
  policy: MultiInputPolicy
  device: cuda
  seed: 45
  learning_rate: 0.0002
  total_timesteps: 3000
  checkpoint_freq: 1000

action_adapter:
  name: raceline_residual_action_adapter
  target_speed_mps: 2.0
  steering_residual_scale: 0.02
  acceleration_residual_scale: 0.02

observation_builder:
  name: raceline_image_speed_observation_builder
```

完全な設定は次のファイルにあります。

```text
aichallenge/ml_workspace/reinforcement_learning/workspace/stable_laps_v4/config.yaml
```

## 10. 実行方法

### 10.1 ビルド

```bash
make autoware-build
```

### 10.2 AWSIMとAutowareの起動

```bash
make dev
```

描画ありAWSIMが起動し、Autoware側では `rl_train` 用の制御構成が起動します。

### 10.3 学習

Autowareコンテナ内で実行します。

```bash
cd /aichallenge/ml_workspace/reinforcement_learning
ROS_DOMAIN_ID=1 python3 ./src/main.py \
  --train \
  --config ./workspace/stable_laps_v4/config.yaml
```

学習済みモデルは設定ファイルと同じディレクトリの `model.zip` に保存されます。チェックポイントは `checkpoints/` に保存されます。

### 10.4 推論評価

```bash
cd /aichallenge/ml_workspace/reinforcement_learning
ROS_DOMAIN_ID=1 python3 ./src/main.py \
  --infer \
  --episodes 1 \
  --model-path ./workspace/stable_laps_v4/model \
  --config ./workspace/stable_laps_v4/config.yaml
```

### 10.5 停止

```bash
make down
```

## 11. シミュレーションを繰り返す際の注意

AWSIMの通常resetだけでは、前回試験後の車両状態が期待どおり初期位置へ戻らないケースがありました。実験条件を揃えるため、重要な学習や評価の前にはAWSIMとAutowareを完全に再起動しました。

```bash
make down
make dev
```

再起動後は、ROSノードとカメラ画像が立ち上がるまで少し待ってから学習・評価を始めます。

## 12. 評価結果

### 学習前の基準制御

残差をゼロに固定した基準制御で、次を確認しました。

- 4,000ステップ連続走行
- 途中リセット0回
- `lap=2` へ2,356ステップで到達
- 最大横ずれ3.42 m
- 最終横ずれ0.26 m
- 最終速度1.58 m/s

### SAC学習

3,000ステップ学習では、1エピソードを途中終了せず学習できました。

- 学習ステップ: 3,000
- エピソード報酬: 約2,390
- SAC更新回数: 2,496

### 学習済みモデルの独立評価

AWSIMとAutowareを完全再起動し、保存したSACモデルを決定論的に評価しました。

- 3,000ステップ完走
- 途中終了なし
- `lap=2` へ2,328ステップで到達
- 最大セクション8
- 最大横ずれ3.35 m
- 最終横ずれ1.31 m
- 最終速度1.45 m/s
- 総報酬2,613.33

この結果から、少なくとも今回の1回の独立評価では、SACを使用した状態で1周以上の安定走行を確認できました。

## 13. テスト

次の処理に単体テストを追加しました。

- 対称なSAC行動からAWSIM用加速値への変換
- 残差制御の操舵方向と速度制御
- セクション番号の周回処理
- 走行ライン観測と前進量
- 自己位置欠損時の誤差抑制
- セクション・ラップ報酬
- 衝突罰則
- 急な操作への罰則

最終確認では8件すべて成功しました。

## 14. 今後の課題

今回確認した独立評価は1エピソードです。「どの実行でも安定している」と判断するには、さらに次の評価が必要です。

1. 複数seedで学習する
2. 各モデルを複数エピソード評価する
3. 完走率、平均横ずれ、最大横ずれ、ラップタイムを集計する
4. 補正幅を少しずつ広げ、性能と安定性の境界を調べる
5. 目標速度を段階的に上げるカリキュラム学習を導入する
6. カメラ画像のタイムアウト頻度を調査する
7. 学習時と推論時のSAC行動分布を記録する
8. コースアウトを速度低下だけでなく横ずれでも終了判定する

まずは現在の低速・小補正設定を安定版として維持し、複数回評価で再現性を確認してから高速化するのが安全です。

## 15. 変更した主なファイル

```text
aichallenge/simulator_scripts/dev.sh
aichallenge/workspace/src/aichallenge_submit/aichallenge_submit_launch/launch/reference.launch.xml
aichallenge/ml_workspace/reinforcement_learning/src/main.py
aichallenge/ml_workspace/reinforcement_learning/src/environment/awsim_env.py
aichallenge/ml_workspace/reinforcement_learning/src/select_parts.py
aichallenge/ml_workspace/reinforcement_learning/src/action/default_action_adapter.py
aichallenge/ml_workspace/reinforcement_learning/src/action/raceline_residual_action_adapter.py
aichallenge/ml_workspace/reinforcement_learning/src/observation/default_observation.py
aichallenge/ml_workspace/reinforcement_learning/src/reward/default_reward.py
aichallenge/ml_workspace/reinforcement_learning/src/context/context_manager.py
aichallenge/ml_workspace/reinforcement_learning/src/context/extract_map/context_extract_map.yaml
aichallenge/ml_workspace/reinforcement_learning/workspace/stable_laps_v4/config.yaml
```

モデル、TensorBoardログ、チェックポイント、AWSIMログなどの生成物は、ソースコードとは分けて管理してください。巨大ファイルや環境固有情報を含む生ログは公開リポジトリへコミットしないよう注意が必要です。
