---
title: AITuberKit passenger commentator demo
date: 2026-08-13
type: tool
source: workspace
series: autonomous-driving-ai-challenge
tags:
  - notes/workspace
  - kind/tool
  - topic/autonomous-driving
  - topic/vlm
  - topic/aituber
  - series/autonomous-driving-ai-challenge
status: draft
---

# AITuberKit passenger commentator demo

AI Challengeの走行実況をAITuberKitへ流し、助手席キャラ風に発話させるための実走デモ用ツール。

## Current Live Path

現行の実走経路は `/speech_event` を使う。

```text
AWSIM / Autoware
  -> tools/rl_metrics/realtime_vlm_commentary_node.py
  -> realtime_commentary.jsonl / vlm_recap_commentary JSONL
  -> tools/aituber_passenger_demo/tail_dual_aituber_commentary_jsonl_measure.py
  -> aituber-server POST /speech_event
  -> AITuberKit external linkage websocket
  -> VOICEVOX speech
```

`/send_message` へ直接送る旧MVPは残しているが、user conversation / critical / VLM recap と同じ優先度制御に乗せる場合は `/speech_event` を使う。

## Scripts

### Live runner

CC1上で公開録画やE2E確認を回す入口。

```bash
bash tools/aituber_passenger_demo/run_step2_dual_lane_aituber_cc1.sh
```

主な設定:

- `RECAP_ENABLED=true`: VLM recapを有効化
- `RECAP_INTERVAL_SEC=12.0`: recap生成間隔
- `RECAP_PREPROCESS=full_320x180`: VLM入力前処理
- `RECAP_GENERATION_DEADLINE_SEC=4.0`: recap生成deadline
- `RECAP_OLLAMA_TIMEOUT_SEC=5.0`: Ollama呼び出しtimeout
- `COURSE_CONTEXT_FILE=/path/to/course_context.json`: 座標ベースのコース文脈
- `PUBLIC_WINDOW_LAYOUT=true`: 公開用画面配置

### JSONL bridge

immediate / VLM recap JSONLを監視し、AITuber serverへ `/speech_event` として送る。

```bash
python3 tools/aituber_passenger_demo/tail_dual_aituber_commentary_jsonl_measure.py \
  --immediate-jsonl /path/to/realtime_commentary.jsonl \
  --recap-jsonl /path/to/vlm_recap_commentary/realtime_commentary.jsonl \
  --endpoint http://127.0.0.1:8018/speech_event
```

bridgeは `event_type` / `commentary_type` / lane から `speechType` を推定し、`priority` と `expireSec` を付けて送る。

## Speech Event Contract

`aituber-server` 側の現行API。

```http
POST /speech_event
Content-Type: application/json

{
  "text": "今の減速、少し強めに入ったね。",
  "speechType": "critical",
  "priority": 100,
  "expireSec": 3.0,
  "source": "vlm_bridge",
  "metadata": {
    "lane": "immediate",
    "event_type": "impact_like"
  }
}
```

優先度の基本方針:

- `critical` / `stuck` / `recovery`: 最優先。user会話中でも通す。
- `user_reply`: ユーザー会話応答。critical系以外には割り込む。
- `course_note`: コース座標由来の補足。
- `vlm_recap`: VLM + Text LLMの周期的な景色/走行recap。
- `immediate` / `normal` / `ambient`: 通常実況。

`user_reply` 送信後は、通常実況系を約2秒だけ抑制する。これは会話応答直後に通常実況がかぶるのを避けるためで、critical/stuck/recoveryは抑制しない。

## Course Context

`COURSE_CONTEXT_FILE` を指定すると、`realtime_vlm_commentary_node.py` が車両位置に近いコース文脈を読み、発話生成の補助情報として使う。

目的は、VLMだけに任せず、コース固有の特徴や見どころを自然に混ぜること。強制的に毎回風景を言わせるのではなく、座標上の文脈としてLLMへ渡す。

## Runtime Deployment

CC1の実走環境では、作業用コピーとして以下にも同じスクリプトを置くことがある。

```text
/home/cc1/aichallenge-work/tail_dual_aituber_commentary_jsonl_measure.py
/home/cc1/aichallenge-work/run_step2_dual_lane_aituber_cc1.sh
```

正式な取り込み先はこの `tools/aituber_passenger_demo/`。実走で直した内容は、ここへ戻してから検証する。

## Legacy MVP

以下は旧MVP用。

- `send_aituber_commentary.py`: 既存JSONLを `/send_message` へ単純送信する。
- `make_phase4_passenger_overlay_cc1.sh`: AITuberKit実画面ではなく、既存動画に助手席UI風overlayを合成する。

旧MVPは見た目確認には使えるが、user conversationの割り込み制御やcritical優先を確認する用途では使わない。
