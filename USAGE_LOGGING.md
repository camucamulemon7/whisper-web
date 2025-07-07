# 使用率ログ機能

## 概要

文字起こしアプリの使用状況を把握するため、セッションごとに以下の情報をログに記録します：

- セッション開始日時
- セッション終了日時
- セッション実行時間
- 音声ソース（screen/microphone/both）
- 使用言語（auto/ja/en）
- バックエンドエンジン（faster-whisper/openai-whisper/whisperx）

## ログフォーマット

### セッション開始時
```
SESSION_START | connection_id=192.168.1.100:12345-1704067200 | start_time=2024-01-01T12:00:00.000000 | total_connections=1
```

### セッション終了時
```
SESSION_END | connection_id=192.168.1.100:12345-1704067200 | end_time=2024-01-01T12:30:00.000000 | duration_seconds=1800.00 | audio_source=screen | language=ja | backend=faster-whisper
```

## ログの確認方法

### 1. Dockerコンテナのログを確認
```bash
# リアルタイムでログを確認
docker compose logs -f backend | grep SESSION_

# 過去のログを確認
docker compose logs backend | grep SESSION_ > session_logs.txt
```

### 2. コンテナ内のログファイルを確認
```bash
# コンテナに入る
docker compose exec backend bash

# ログを確認
cat /app/logs/app.log | grep SESSION_
```

## 使用統計の分析

### analyze_usage.pyスクリプトの使用

付属の`analyze_usage.py`スクリプトを使用して、ログから統計情報を生成できます。

```bash
# ログをエクスポート
docker compose logs backend > backend.log

# 実行権限を付与（初回のみ）
chmod +x backend/analyze_usage.py

# 統計を分析
python backend/analyze_usage.py backend.log
# または
./backend/analyze_usage.py backend.log
```

### 出力例
```
================================================================================
WHISPER REAL-TIME TRANSCRIPTION - USAGE STATISTICS
================================================================================

📊 OVERALL STATISTICS
  Total Sessions: 25
  Total Duration: 15h 30m 45s
  Average Duration: 37m 13s

📅 DAILY USAGE
  2024-01-01: 10 sessions, 6h 15m 30s
  2024-01-02: 8 sessions, 5h 10m 15s
  2024-01-03: 7 sessions, 4h 5m 0s

🎤 BY AUDIO SOURCE
  screen: 18 sessions (72.0%), 11h 20m 30s
  microphone: 5 sessions (20.0%), 2h 45m 15s
  both: 2 sessions (8.0%), 1h 25m 0s

🌐 BY LANGUAGE
  ja: 15 sessions (60.0%), 9h 30m 45s
  auto: 8 sessions (32.0%), 4h 50m 0s
  en: 2 sessions (8.0%), 1h 10m 0s

⚙️  BY BACKEND
  faster-whisper: 23 sessions (92.0%), 14h 15m 45s
  openai-whisper: 2 sessions (8.0%), 1h 15m 0s

📈 NOTABLE SESSIONS
  Longest: 2h 30m 15s on 2024-01-02
  Shortest: 2m 30s on 2024-01-03

================================================================================
```

## カスタム分析

ログデータはシンプルな形式なので、以下のようなカスタム分析も可能です：

### 時間帯別の使用状況
```bash
docker compose logs backend | grep SESSION_START | awk -F'start_time=' '{print $2}' | cut -d'T' -f2 | cut -d':' -f1 | sort | uniq -c
```

### 特定の日付の使用状況
```bash
docker compose logs backend | grep SESSION_ | grep "2024-01-01"
```

### 平均セッション時間の計算
```bash
docker compose logs backend | grep SESSION_END | awk -F'duration_seconds=' '{print $2}' | awk '{sum+=$1; count++} END {print "Average:", sum/count, "seconds"}'
```

## ログのローテーション

長期運用する場合は、ログのローテーションを設定することをお勧めします：

```yaml
# docker-compose.yml に追加
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
```

## プライバシーへの配慮

- IPアドレスは記録されますが、個人を特定する情報は含まれません
- 音声データや文字起こし内容は一切記録されません
- 必要に応じて、IPアドレスをハッシュ化することも可能です
