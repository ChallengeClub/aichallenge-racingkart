# VLM走行実況ツール

このディレクトリには、自動運転AIチャレンジ / AWSIM の車載カメラ映像とROS 2 bagを使って、VLMによる走行実況を生成するための実験用ツール群があります。

目的は、走行動画をあとから振り返りやすくしたり、VLMを自動運転開発の観察・説明・デバッグ補助に使えるかを試すことです。公式評価の実行に必要なコードではありません。

## 関連資料

- 現時点のまとめ記事: https://qiita.com/kiwsdiv/items/94ff578a486f022119f0
- 走行動画: https://www.youtube.com/watch?v=o4JYSiev4UI

## できること

- 録画済みの車載カメラ動画とrosbag/mcapを同期して、オフラインで走行実況を生成する
- ROS 2 topicを購読しながら、準リアルタイムに実況テキストを生成する
- VOICEVOXを使って実況音声を生成する
- 生成した音声を動画に重ね、遅延やキュー詰まりを可視化する
- VLMに渡すカメラ画像の前処理方法を比較する

現時点では、Ollama上の `llava:7b` を画像認識、`qwen3:8b` を日本語文生成に使う構成を主に想定しています。低遅延を優先する場合は、VLMを使わずテンプレートで実況するモードも用意しています。

## 主要スクリプト

| ファイル | 役割 |
| --- | --- |
| `synced_mcap_vlm_commentary.py` | 録画済み動画とrosbag/mcapを同期して実況を生成する |
| `realtime_vlm_commentary_node.py` | カメラ画像と車両状態topicを購読して準リアルタイム実況を生成するROS 2 node |
| `run_mpc_camera_rosbag_capture.sh` | MPC走行、カメラ録画、rosbag記録、任意でリアルタイム実況をまとめて実行する補助スクリプト |
| `make_realtime_commentary_mix.py` | realtime nodeが出力したWAVを動画に重ね、遅延表示つき動画を作る |
| `make_dual_realtime_commentary_mix.py` | 運転実況と風景実況の2系統音声を動画に重ねる |
| `make_voicevox_commentary_audio.py` | `commentary.jsonl` からVOICEVOX音声WAVを作る |
| `compare_camera_preprocessing_vlm.py` | カメラ前処理の違いによるVLM実況結果を比較する |
| `benchmark_vlm_preprocessing_tags.py` | カメラ前処理ごとのVLMタグ抽出をベンチマークする |
| `sensor_grounded_vlm_commentary.py` | カメラフレームと車両データを使って、センサ情報に基づく実況を生成する |
| `ros_camera_viewer.py` | ROS 2 camera topicを簡易表示する |

## 必要な環境

基本的には、自動運転AIチャレンジの開発環境内で使う想定です。ROS 2 HumbleとAI Challenge workspaceが使える状態を前提にしています。

追加で使うもの:

- Python: `opencv-python`, `numpy`
- ROS Python: `rclpy`, `cv_bridge`, `rosbag2_py`, `rosidl_runtime_py`
- ローカルVLMサーバ: Ollama
- 任意: VOICEVOX
- 任意: ffmpeg、または `linuxserver/ffmpeg:latest` コンテナ

Ollamaモデルの例:

```bash
ollama pull llava:7b
ollama pull qwen3:8b
```

VOICEVOXは必須ではありません。VOICEVOXなしでも、実況テキストのJSONL出力までは生成できます。

## まず試すなら: 録画済み動画からのオフライン実況生成

チーム内で最初に共有・再現するなら、リアルタイムnodeよりもオフライン生成のほうが扱いやすいです。

```bash
python3 tools/rl_metrics/synced_mcap_vlm_commentary.py \
  --video output/<run_id>/d1/capture/<capture>.mp4 \
  --bag output/<run_id>/d1/rosbag2_autoware \
  --output-dir output/vlm-commentary/<run_id> \
  --times 5,15,25,35,45,55 \
  --preprocess lower_half_160x80 \
  --vision-model llava:7b \
  --text-model qwen3:8b
```

主な出力:

- `commentary.jsonl`
- 同期した車両データ
- サンプリングしたカメラフレーム
- 生成された実況テキスト

動画開始時刻とbag開始時刻にずれがある場合は、以下で補正します。

```bash
--bag-video-offset-sec <seconds>
```

## VOICEVOXで実況音声を作る

`commentary.jsonl` を生成したあと、VOICEVOXで音声WAVを作れます。

```bash
python3 tools/rl_metrics/make_voicevox_commentary_audio.py \
  --commentary-jsonl output/vlm-commentary/<run_id>/commentary.jsonl \
  --output-wav output/vlm-commentary/<run_id>/commentary.wav \
  --manifest output/vlm-commentary/<run_id>/voicevox_manifest.json \
  --voicevox-url http://127.0.0.1:50021 \
  --speaker 3
```

## 準リアルタイム実況node

`realtime_vlm_commentary_node.py` は、カメラ画像と車両状態topicを購読しながら実況を生成するROS 2 nodeです。

VLMを使うモード、テンプレートだけで生成するモード、VOICEVOXで音声化するモードがあります。

テンプレートのみの低遅延例:

```bash
python3 tools/rl_metrics/realtime_vlm_commentary_node.py \
  --image-topic /sensing/camera/image_raw \
  --odom-topic /localization/kinematic_state \
  --control-topic /control/command/control_cmd \
  --accel-topic /localization/acceleration \
  --output-dir /tmp/realtime_vlm_commentary \
  --interval-sec 3.0 \
  --preprocess lower_80x40 \
  --commentary-mode template \
  --template-style event_fast
```

VLMとLLMを使う例:

```bash
python3 tools/rl_metrics/realtime_vlm_commentary_node.py \
  --output-dir /tmp/realtime_vlm_commentary \
  --interval-sec 3.0 \
  --preprocess lower_80x40 \
  --commentary-mode llm \
  --vision-model llava:7b \
  --text-model qwen3:8b
```

VOICEVOXで音声も生成する例:

```bash
python3 tools/rl_metrics/realtime_vlm_commentary_node.py \
  --output-dir /tmp/realtime_vlm_commentary \
  --commentary-mode template \
  --template-style event_fast \
  --voicevox-url http://127.0.0.1:50021 \
  --voicevox-speaker 3 \
  --voicevox-speed-scale 1.30
```

低遅延で使う場合のおすすめ設定:

- `--commentary-mode template`
- `--template-style event_fast`
- `--commentary-trigger event`
- `--min-speak-interval-sec 3.0`
- `--voicevox-speed-scale 1.30` 前後

毎回VLM推論を待たないため、実況タイミングを優先できます。

## 走行・録画・実況をまとめて実行する

`run_mpc_camera_rosbag_capture.sh` は、MPC走行、カメラ録画、rosbag記録、任意でリアルタイム実況node起動までまとめて行う補助スクリプトです。

例:

```bash
RUN_ID=mpc-vlm-commentary-demo \
REALTIME_COMMENTARY=true \
REALTIME_COMMENTARY_MODE=template \
REALTIME_COMMENTARY_TRIGGER=event \
REALTIME_TEMPLATE_STYLE=event_fast \
REALTIME_VOICEVOX_URL=http://127.0.0.1:50021 \
REALTIME_VOICEVOX_SPEED_SCALE=1.30 \
tools/rl_metrics/run_mpc_camera_rosbag_capture.sh
```

主な環境変数:

| 変数 | 既定値 | 意味 |
| --- | --- | --- |
| `REALTIME_COMMENTARY` | `false` | リアルタイム実況nodeを起動する |
| `REALTIME_IMAGE_TOPIC` | `/sensing/camera/image_raw` | カメラtopic |
| `REALTIME_INTERVAL_SEC` | `3.0` | 実況生成間隔 |
| `REALTIME_PREPROCESS` | `lower_80x40` | VLM入力用の画像前処理 |
| `REALTIME_COMMENTARY_MODE` | `llm` | `llm` または `template` |
| `REALTIME_COMMENTARY_TRIGGER` | `interval` | `interval` または `event` |
| `REALTIME_TEMPLATE_STYLE` | `normal` | `normal`, `short`, `event`, `event_fast` など |
| `REALTIME_VISION_MODEL` | `llava:7b` | Ollamaの画像モデル |
| `REALTIME_TEXT_MODEL` | `qwen3:8b` | Ollamaのテキストモデル |
| `REALTIME_VOICEVOX_URL` | 空 | 指定するとVOICEVOX音声を生成する |
| `REALTIME_VOICEVOX_SPEAKER` | `3` | VOICEVOX speaker id |
| `REALTIME_VOICEVOX_SPEED_SCALE` | `1.08` | 話速 |

## 生成した音声を動画に重ねる

realtime nodeで生成したWAVを、録画済み動画に重ねる例です。

```bash
python3 tools/rl_metrics/make_realtime_commentary_mix.py \
  output/<run_id>/d1 \
  --video output/<run_id>/d1/capture/<capture>.mp4 \
  --timeline-start-wall-file output/<run_id>/d1/logs/capture_start_wall_time_sec.txt \
  --start-mode audio_ready \
  --queue \
  --drop-stale-sec 4.0 \
  --output-mp4 output/<run_id>/d1/capture/<capture>_vlm_commentary.mp4 \
  --schedule-jsonl output/<run_id>/d1/realtime_commentary_schedule.jsonl
```

`--timeline-start-wall-file` がある場合は指定してください。画面録画の開始wall-timeを基準に音声を配置できるため、動画と実況の同期が取りやすくなります。

## 出力ファイル

realtime nodeの代表的な出力は以下です。

```text
realtime_commentary.jsonl
realtime_commentary.md
audio/
  line_000.wav
  line_001.wav
```

JSONLには、以下のような情報が入ります。

- 画像timestamp
- 車速
- 制御指令
- 生成テキスト
- VLM / テキスト生成 / TTS の遅延
- 音声ファイルパス
- wall-clock timing

## 現時点の制限

- 実験用ツールであり、公式評価用の本番コードではない
- VLMの遅延は、モデルサイズ、GPU/CPU、画像前処理に大きく依存する
- リアルタイム性を優先するなら、LLM生成よりテンプレート生成のほうが安定する
- VOICEVOX音声が長すぎるとキューが詰まり、実況が遅れる
- カメラ前処理はAIチャレンジの車載カメラ視点に合わせている
- 他のカメラ視点や別コースでは前処理の調整が必要
- このブランチは公式評価の挙動を変更しない

## チーム内での共有手順案

1. まず `synced_mcap_vlm_commentary.py` で録画済み動画から実況を作る
2. `commentary.jsonl` と実況つき動画を共有する
3. 次に `realtime_vlm_commentary_node.py` をテンプレート/eventモードで試す
4. VLM推論は、最初の必須依存ではなく、追加機能として試す

