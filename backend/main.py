"""
FastAPI WebSocket server for real-time speech transcription using Whisper
リアルタイム音声文字起こしのためのFastAPI WebSocketサーバー
"""

import asyncio
import json
import logging
import os
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stt import TranscriptionProcessor
from stt_openai import OpenAIWhisperProcessor
from stt_whisperx import WhisperXProcessor
import httpx
from dotenv import load_dotenv
import pynvml

# .envファイルを読み込み
load_dotenv()

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper Real-time Transcription")

# CORS設定（開発用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# アクティブなWebSocket接続を管理
active_connections: Set[WebSocket] = set()

# Pydanticモデル
class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str

class TranscriptionItem(BaseModel):
    text: str
    timestamp: int

class CorrectRequest(BaseModel):
    transcriptions: list[TranscriptionItem]

class CorrectionItem(BaseModel):
    timestamp: int
    corrected_text: str

class CorrectResponse(BaseModel):
    corrections: list[CorrectionItem]

class StatsResponse(BaseModel):
    active_connections: int
    status: str
    whisper_model: str
    whisper_device: str
    gpu_vram_used_gb: float | None = None
    gpu_vram_total_gb: float | None = None
    gpu_vram_usage_percent: float | None = None
    gpu_name: str | None = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "service": "whisper-realtime-stt"}

@app.websocket("/ws/stt")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocketエンドポイント: 音声ストリームを受信し、文字起こし結果を返す
    
    Args:
        websocket: FastAPI WebSocket instance
    """
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"New WebSocket connection. Total connections: {len(active_connections)}")
    
    # 文字起こしプロセッサーの初期化（デフォルトでfaster-whisper）
    processor = None
    backend_type = 'faster-whisper'
    
    try:
        # プロセッサーを初期化
        processor = TranscriptionProcessor()
        logger.info("Processor initialized successfully")
        # 音声データ受信と文字起こし処理のループ
        while True:
            try:
                # メッセージを受信
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
            
            if "text" in message:
                # テキストメッセージ（設定など）
                data = json.loads(message["text"])
                if data.get("type") == "config":
                    # 言語設定を更新
                    language = data.get("language")
                    processor.set_language(language)
                    logger.info(f"Language set to: {language or 'auto'}")
                    
                    # バックエンドの変更が要求された場合
                    new_backend = data.get("backend")
                    if new_backend and new_backend != backend_type:
                        # 古いプロセッサーをクリーンアップ
                        await processor.cleanup()
                        
                        # 新しいプロセッサーを作成
                        if new_backend == 'openai-whisper':
                            processor = OpenAIWhisperProcessor()
                            backend_type = 'openai-whisper'
                            logger.info("Switched to OpenAI Whisper backend")
                        elif new_backend == 'whisperx':
                            # WhisperXのサポート
                            try:
                                processor = WhisperXProcessor()
                                backend_type = 'whisperx'
                                logger.info("Switched to WhisperX backend")
                            except Exception as e:
                                logger.error(f"Failed to initialize WhisperX: {e}")
                                logger.warning("Falling back to faster-whisper")
                                processor = TranscriptionProcessor()
                                backend_type = 'faster-whisper'
                        else:
                            processor = TranscriptionProcessor()
                            backend_type = 'faster-whisper'
                            logger.info("Switched to Faster Whisper backend")
                        
                        # 言語設定を再適用
                        processor.set_language(language)
                    
                    # パラメータの更新
                    parameters = data.get("parameters")
                    if parameters:
                        processor.update_parameters(parameters)
                        logger.info(f"Updated parameters: {parameters}")
                    
                    continue
            
            elif "bytes" in message:
                # バイナリデータ（PCM音声）
                audio_data = message["bytes"]
                logger.debug(f"Received audio chunk: {len(audio_data)} bytes")
                
                # 音声データを処理し、文字起こし結果を取得
                transcriptions = await processor.process_audio_chunk(audio_data)
                
                # 文字起こし結果をクライアントに送信
                for transcription in transcriptions:
                    try:
                        await websocket.send_json(transcription)
                        logger.info(f"Sent transcription: {transcription}")
                    except WebSocketDisconnect:
                        logger.info("WebSocket disconnected during send")
                        raise
                    except Exception as e:
                        logger.error(f"Error sending transcription: {e}")
                        raise
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
    finally:
        # クリーンアップ
        if websocket in active_connections:
            active_connections.remove(websocket)
        if processor:
            try:
                await processor.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        logger.info(f"Connection closed. Remaining connections: {len(active_connections)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    接続統計情報を返すエンドポイント
    """
    stats = {
        "active_connections": len(active_connections),
        "status": "healthy",
        "whisper_model": os.getenv('WHISPER_MODEL', 'large-v3'),
        "whisper_device": os.getenv('WHISPER_DEVICE', 'cuda')
    }
    
    # GPU情報を取得（CUDAが利用可能な場合のみ）
    device = os.getenv('WHISPER_DEVICE', 'cuda')
    logger.info(f"Checking GPU stats for device: {device}")
    
    if device == 'cuda':
        try:
            logger.info("Attempting to get GPU information...")
            # NVMLを初期化
            pynvml.nvmlInit()
            
            # GPU 0の情報を取得
            # Docker内ではCUDA_VISIBLE_DEVICESに関係なく、常にindex 0を使用
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU名を取得
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            
            # メモリ情報を取得
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_gb = mem_info.used / (1024 ** 3)  # バイトからGBに変換
            vram_total_gb = mem_info.total / (1024 ** 3)
            vram_usage_percent = (mem_info.used / mem_info.total) * 100
            
            stats.update({
                "gpu_vram_used_gb": round(vram_used_gb, 2),
                "gpu_vram_total_gb": round(vram_total_gb, 2),
                "gpu_vram_usage_percent": round(vram_usage_percent, 1),
                "gpu_name": gpu_name
            })
            
            logger.info(f"GPU stats retrieved successfully: {gpu_name}, {vram_used_gb:.2f}/{vram_total_gb:.2f}GB ({vram_usage_percent:.1f}%)")
            
            # NVMLをクリーンアップ
            pynvml.nvmlShutdown()
            
        except pynvml.NVMLError as e:
            logger.error(f"NVML Error getting GPU stats: {e}")
            logger.error(f"NVML Error code: {e.value if hasattr(e, 'value') else 'unknown'}")
        except Exception as e:
            logger.error(f"Unexpected error getting GPU stats: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.info(f"GPU stats not available - device is '{device}', not 'cuda'")
    
    return StatsResponse(**stats)

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    LLMを使用してテキストの要約を返す
    """
    # 環境変数から設定を取得
    api_key = os.getenv('LLM_API_KEY')
    model = os.getenv('LLM_MODEL', 'gpt-4')
    base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
    
    # API keyがない場合はモックレスポンスを返す
    if not api_key or api_key == 'your-api-key-here':
        logger.warning("LLM API key not configured, returning mock response")
        await asyncio.sleep(0.3)  # 300msの遅延をシミュレート
        return SummarizeResponse(summary="(summary)")
    
    try:
        # LLM APIを呼び出し
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "あなたは優秀な要約アシスタントです。与えられたテキストを簡潔に要約してください。重要なポイントを箇条書きで3-5点にまとめてください。"
                        },
                        {
                            "role": "user",
                            "content": f"以下のテキストを要約してください：\n\n{request.text}"
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30.0
            )
            
            response.raise_for_status()
            result = response.json()
            summary = result['choices'][0]['message']['content']
            
            return SummarizeResponse(summary=summary)
            
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        return SummarizeResponse(summary="要約の生成に失敗しました。")

@app.post("/api/correct", response_model=CorrectResponse)
async def correct_transcriptions(request: CorrectRequest):
    """
    LLMを使用して文字起こしの誤字を修正する
    """
    # 環境変数から設定を取得
    api_key = os.getenv('LLM_API_KEY')
    model = os.getenv('LLM_MODEL', 'gpt-4')
    base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
    
    # API keyがない場合はモックレスポンスを返す
    if not api_key or api_key == 'your-api-key-here':
        logger.warning("LLM API key not configured, returning original text")
        return CorrectResponse(corrections=[
            CorrectionItem(timestamp=t.timestamp, corrected_text=t.text) 
            for t in request.transcriptions
        ])
    
    try:
        # 文字起こしテキストを結合
        texts = [t.text for t in request.transcriptions]
        combined_text = "\n".join(texts)
        
        # LLM APIを呼び出し
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "あなたは優秀な校正者です。音声認識による文字起こしの誤字・脱字を修正してください。各行ごとに修正し、行の区切りは維持してください。意味を変えないよう注意し、明らかな誤字・脱字のみを修正してください。"
                        },
                        {
                            "role": "user",
                            "content": f"以下の文字起こしを修正してください。各行ごとに修正し、行数は変えないでください：\n\n{combined_text}"
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=30.0
            )
            
            response.raise_for_status()
            result = response.json()
            corrected_text = result['choices'][0]['message']['content']
            
            # 修正されたテキストを行ごとに分割
            corrected_lines = corrected_text.strip().split('\n')
            
            # タイムスタンプと修正テキストをペアにする
            corrections = []
            for i, (trans, corrected) in enumerate(zip(request.transcriptions, corrected_lines)):
                corrections.append(CorrectionItem(
                    timestamp=trans.timestamp,
                    corrected_text=corrected.strip()
                ))
            
            return CorrectResponse(corrections=corrections)
            
    except Exception as e:
        logger.error(f"Error calling LLM API for correction: {e}")
        # エラーの場合は元のテキストを返す
        return CorrectResponse(corrections=[
            CorrectionItem(timestamp=t.timestamp, corrected_text=t.text) 
            for t in request.transcriptions
        ])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=os.getenv('API_HOST', '0.0.0.0'), 
        port=int(os.getenv('API_PORT', '8000'))
    )
