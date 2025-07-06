"""
Speech-to-Text processing module using WhisperX 3.4.2
WhisperX 3.4.2を使用した音声文字起こし処理モジュール
"""

import asyncio
import numpy as np
import logging
import time
from typing import List, Dict, Optional
from collections import deque
import os
import threading
import queue
import torch
import struct
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class WhisperXProcessor:
    """
    音声データを処理し、WhisperXで文字起こしを行うクラス
    WhisperX 3.4.2は高精度な音声認識と話者分離機能を提供
    """
    
    def __init__(self):
        """
        環境変数から設定を読み込んで初期化
        """
        self.language = None  # None = 自動検出
        # 環境変数から設定を読み込み
        self.model_size = os.getenv('WHISPER_MODEL', 'large-v3')
        self.device = os.getenv('WHISPER_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.compute_type = os.getenv('WHISPER_COMPUTE_TYPE', 'float16' if self.device == 'cuda' else 'int8')
        
        # WhisperXパラメータ
        self.batch_size = int(os.getenv('WHISPERX_BATCH_SIZE', '16'))
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.temperature = float(os.getenv('WHISPER_TEMPERATURE', '0.0'))
        self.prompt = os.getenv('WHISPER_PROMPT', None) or None
        self.chunk_duration = 10  # 10秒チャンク
        
        # WhisperXモデルの初期化
        logger.info(f"Loading WhisperX model: {self.model_size} on {self.device}")
        try:
            import whisperx
            
            # WhisperXモデルのロード
            self.model = whisperx.load_model(
                self.model_size, 
                self.device, 
                compute_type=self.compute_type,
                language=self.language  # 自動検出の場合はNone
            )
            
            # アライメントモデルのロード（言語が指定されている場合）
            self.align_model = None
            self.metadata = None
            if self.language:
                self.align_model, self.metadata = whisperx.load_align_model(
                    language_code=self.language, 
                    device=self.device
                )
            
            logger.info("WhisperX model loaded successfully")
        except ImportError:
            logger.error("WhisperX is not installed. Please install it with: pip install whisperx==3.4.2")
            # フォールバック：通常のWhisperを使用
            logger.warning("Falling back to regular Whisper model")
            import whisper
            self.model = whisper.load_model(self.model_size)
            self.align_model = None
            self.metadata = None
            self.use_whisperx = False
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise
        else:
            self.use_whisperx = True
        
        # 音声バッファ（15秒分のPCMデータを保持）
        self.audio_buffer = deque(maxlen=15 * 16000)
        
        # 処理用のキューとスレッド
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 文字起こし履歴（重複検出用）
        self.transcription_history = deque(maxlen=20)
        
        # 処理スレッドを開始
        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 最後の文字起こし時刻
        self.last_transcription_time = 0
        self.min_transcription_interval = 2.0  # 最小間隔（秒）
        
        logger.info("WhisperXProcessor initialized")
    
    def _process_audio_loop(self):
        """
        音声処理ループ（別スレッドで実行）
        """
        accumulated_audio = []
        accumulated_duration = 0
        
        while True:
            try:
                # キューから音声データを取得（タイムアウト付き）
                audio_data = self.audio_queue.get(timeout=1.0)
                
                if audio_data is None:
                    # 終了シグナル
                    break
                
                # 音声データを蓄積
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                accumulated_audio.append(audio_array)
                accumulated_duration += len(audio_array) / 16000.0
                
                # 10秒分のデータが蓄積されたら処理
                if accumulated_duration >= self.chunk_duration:
                    combined_audio = np.concatenate(accumulated_audio)
                    self._transcribe_audio(combined_audio)
                    
                    # バッファをクリア
                    accumulated_audio = []
                    accumulated_duration = 0
                    
            except queue.Empty:
                # タイムアウト時、蓄積されたデータがあれば処理
                if accumulated_audio and accumulated_duration > 1.0:
                    combined_audio = np.concatenate(accumulated_audio)
                    self._transcribe_audio(combined_audio)
                    accumulated_audio = []
                    accumulated_duration = 0
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def _transcribe_audio(self, audio_data: np.ndarray):
        """
        音声データを文字起こし
        """
        # 音量チェック
        if np.max(np.abs(audio_data)) < 0.001:
            logger.debug("Audio is too quiet, skipping transcription")
            return
        
        try:
            if self.use_whisperx:
                import whisperx
                
                # WhisperXで文字起こし
                result = self.model.transcribe(
                    audio_data,
                    batch_size=self.batch_size,
                    language=self.language,
                    temperature=self.temperature,
                    initial_prompt=self.prompt
                )
                
                # アライメント（言語が指定されている場合）
                if self.align_model and result["segments"]:
                    result = whisperx.align(
                        result["segments"], 
                        self.align_model, 
                        self.metadata, 
                        audio_data, 
                        self.device,
                        return_char_alignments=False
                    )
                
                # セグメントを処理
                current_time = time.time()
                for segment in result.get("segments", []):
                    text = segment.get("text", "").strip()
                    if not text:
                        continue
                    
                    # 重複チェック
                    if self._is_duplicate_or_similar(text):
                        continue
                    
                    # 履歴に追加
                    self.transcription_history.append(text)
                    
                    transcription = {
                        "text": text,
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "is_final": True,
                        "timestamp": int(current_time * 1000)  # ミリ秒に変換
                    }
                    
                    # 話者情報があれば追加
                    if "speaker" in segment:
                        transcription["speaker"] = segment["speaker"]
                    
                    self.result_queue.put(transcription)
                    logger.info(f"Transcribed: {text}")
                    
            else:
                # 通常のWhisperで文字起こし（フォールバック）
                result = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    temperature=self.temperature,
                    fp16=self.device == "cuda"
                )
                
                text = result['text'].strip()
                if text and not self._is_duplicate_or_similar(text):
                    self.transcription_history.append(text)
                    
                    current_time = time.time()
                    transcription = {
                        "text": text,
                        "start": 0.0,
                        "end": len(audio_data) / 16000.0,
                        "is_final": True,
                        "timestamp": int(current_time * 1000)  # ミリ秒に変換
                    }
                    
                    self.result_queue.put(transcription)
                    logger.info(f"Transcribed: {text}")
                    
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _is_duplicate_or_similar(self, text: str) -> bool:
        """
        テキストが重複または類似しているかチェック
        """
        if not text.strip():
            return True
        
        # 短いテキストの重複チェック
        if len(text) < 10:
            return text in self.transcription_history
        
        # 完全一致チェック
        for hist_text in self.transcription_history:
            if text == hist_text:
                logger.info(f"Duplicate found: {text}")
                return True
            
            # 類似度チェック（編集距離ベース）
            if len(hist_text) > 10:
                similarity = SequenceMatcher(None, text.lower(), hist_text.lower()).ratio()
                if similarity > 0.85:
                    logger.info(f"Similar text found: {text} ~ {hist_text} (similarity: {similarity:.2f})")
                    return True
        
        return False
    
    async def process_audio_chunk(self, audio_data: bytes) -> List[Dict]:
        """
        音声チャンクを処理し、文字起こし結果を返す
        """
        # 音声データをキューに追加
        self.audio_queue.put(audio_data)
        
        # 結果を収集
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def set_language(self, language: Optional[str]):
        """
        言語設定を更新
        """
        self.language = language
        logger.info(f"Language set to: {language or 'auto'}")
        
        # アライメントモデルを再ロード（WhisperXの場合）
        if self.use_whisperx and language:
            try:
                import whisperx
                self.align_model, self.metadata = whisperx.load_align_model(
                    language_code=language, 
                    device=self.device
                )
                logger.info(f"Alignment model loaded for language: {language}")
            except Exception as e:
                logger.warning(f"Failed to load alignment model: {e}")
                self.align_model = None
                self.metadata = None
    
    def update_parameters(self, parameters: Dict):
        """
        WhisperXパラメータを更新
        
        Args:
            parameters: 更新するパラメータの辞書
        """
        # promptの更新
        if 'prompt' in parameters:
            self.prompt = parameters['prompt'] or None
        
        # batch_sizeの更新
        if 'beam_size' in parameters:
            # WhisperXではbatch_sizeとして扱う
            self.batch_size = int(parameters['beam_size'])
        
        # chunk_durationの更新
        if 'chunk_duration' in parameters:
            self.chunk_duration = int(parameters['chunk_duration'])
        
        # temperatureの更新
        if 'temperature' in parameters:
            self.temperature = float(parameters['temperature'])
        
        logger.info(f"Parameters updated: {parameters}")
    
    async def cleanup(self):
        """
        リソースのクリーンアップ
        """
        # 処理スレッドを停止
        self.audio_queue.put(None)
        self.processing_thread.join(timeout=5)
        
        logger.info("WhisperXProcessor cleaned up")
