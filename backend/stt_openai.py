"""
Speech-to-Text processing module using OpenAI Whisper
OpenAI Whisperを使用した音声文字起こし処理モジュール
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
import whisper
import torch

logger = logging.getLogger(__name__)

class OpenAIWhisperProcessor:
    """
    音声データを処理し、OpenAI Whisperで文字起こしを行うクラス
    """
    
    def __init__(self):
        """
        環境変数から設定を読み込んで初期化
        """
        self.language = None  # None = 自動検出
        
        # 環境変数から設定を読み込み
        self.model_size = os.getenv('WHISPER_MODEL', 'large-v3')
        self.device = os.getenv('WHISPER_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Whisperモデルの初期化
        logger.info(f"Loading OpenAI Whisper model: {self.model_size} on {self.device}")
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # 音声バッファ（15秒分のPCMデータを保持）
        self.audio_buffer = deque(maxlen=15 * 16000)  # 15秒 * 16kHz
        
        # 重複防止用の文字起こし履歴
        self.transcription_history = deque(maxlen=10)
        self.last_transcription = ""
        self.last_transcription_time = 0
        
        # 無音検出用
        self.silence_duration = 0
        self.silence_threshold = 0.01
        self.silence_clear_duration = 3.0
        self.min_audio_length = 1.0
        
        # チャンク処理設定
        self.chunk_duration = 10.0  # 10秒のチャンクで処理
        self.overlap_duration = 1.0  # 1秒のオーバーラップ
        
        # 処理用スレッドとキュー
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_audio_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 最後の処理時刻
        self.last_transcribe_time = time.time()
        self.last_audio_time = time.time()
        
        logger.info("OpenAIWhisperProcessor initialized")
    
    def _process_audio_thread(self):
        """
        音声処理用のワーカースレッド
        """
        total_bytes_received = 0
        chunk_count = 0
        
        while True:
            try:
                # キューから音声データを取得
                audio_data = self.audio_queue.get()
                if audio_data is None:  # 終了シグナル
                    break
                
                chunk_count += 1
                total_bytes_received += len(audio_data)
                logger.info(f"Received PCM chunk #{chunk_count}: {len(audio_data)} bytes")
                
                # PCMデータ（Int16）をFloat32に変換
                try:
                    pcm_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array = pcm_int16.astype(np.float32) / 32768.0
                    
                    logger.info(f"Converted PCM data: {len(audio_data)} bytes -> {len(audio_array)} samples")
                    
                    # 無音検出
                    max_amplitude = np.max(np.abs(audio_array))
                    current_time = time.time()
                    
                    if max_amplitude < self.silence_threshold:
                        # 無音の場合
                        self.silence_duration += current_time - self.last_audio_time
                        if self.silence_duration > self.silence_clear_duration:
                            logger.info(f"Clearing buffer after {self.silence_duration:.1f}s of silence")
                            self.audio_buffer.clear()
                            self.transcription_history.clear()
                            self.silence_duration = 0
                    else:
                        # 音声がある場合
                        self.silence_duration = 0
                        self.audio_buffer.extend(audio_array)
                        logger.info(f"Audio buffer size: {len(self.audio_buffer)} samples")
                    
                    self.last_audio_time = current_time
                    
                    # 定期的に文字起こしを実行
                    buffer_duration = len(self.audio_buffer) / 16000.0
                    
                    should_process = False
                    if buffer_duration >= self.chunk_duration:
                        should_process = True
                        logger.info(f"Processing full chunk: {buffer_duration:.1f}s")
                    elif max_amplitude < self.silence_threshold and buffer_duration >= self.min_audio_length:
                        if current_time - self.last_transcribe_time >= 2.0:
                            should_process = True
                            logger.info(f"Processing on silence: {buffer_duration:.1f}s")
                    
                    if should_process:
                        self._transcribe_buffer()
                        self.last_transcribe_time = current_time
                        
                        # オーバーラップ分を残してバッファをクリア
                        if len(self.audio_buffer) > self.overlap_duration * 16000:
                            overlap_samples = int(self.overlap_duration * 16000)
                            new_buffer = deque(maxlen=15 * 16000)
                            new_buffer.extend(list(self.audio_buffer)[-overlap_samples:])
                            self.audio_buffer = new_buffer
                            logger.info(f"Kept {overlap_samples} samples for overlap")
                        else:
                            self.audio_buffer.clear()
                        
                except Exception as e:
                    logger.error(f"Error processing PCM data: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                        
            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def _transcribe_buffer(self):
        """
        バッファ内の音声を文字起こし
        """
        if len(self.audio_buffer) < 16000:  # 1秒未満の場合はスキップ
            logger.warning(f"Buffer too small for transcription: {len(self.audio_buffer)} samples")
            return
        
        # バッファから音声データを取得（最大10秒）
        audio_data = np.array(list(self.audio_buffer))
        logger.info(f"Transcribing audio: {len(audio_data)} samples")
        
        # 無音チェック
        if np.max(np.abs(audio_data)) < 0.003:
            logger.info("Audio appears to be silence, skipping transcription")
            return
        
        try:
            # OpenAI Whisperで文字起こし
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                temperature=0.0,
                no_speech_threshold=0.6,
                fp16=torch.cuda.is_available()
            )
            
            text = result['text'].strip()
            if not text:
                return
            
            # 重複チェック
            if self._is_duplicate(text):
                return
            
            # 履歴に追加
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
            
            self.last_transcription = text
            self.last_transcription_time = current_time
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _is_duplicate(self, text: str) -> bool:
        """
        テキストが重複しているかチェック
        """
        # 空のテキストはスキップ
        if not text.strip():
            return True
        
        # 完全一致チェック
        for hist_text in self.transcription_history:
            if text == hist_text:
                logger.info(f"Duplicate found: {text}")
                return True
            
            # 部分文字列チェック
            if len(text) > 10 and len(hist_text) > 10:
                if text in hist_text or hist_text in text:
                    logger.info(f"Substring found: {text} in {hist_text}")
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
    
    def update_parameters(self, parameters: Dict):
        """
        Whisperパラメータを更新
        
        Args:
            parameters: 更新するパラメータの辞書
        """
        # OpenAI Whisperでは一部のパラメータのみサポート
        if 'prompt' in parameters:
            self.prompt = parameters['prompt'] or None
        
        if 'temperature' in parameters:
            self.temperature = float(parameters['temperature'])
        
        if 'chunk_duration' in parameters:
            self.chunk_duration = int(parameters['chunk_duration'])
        
        logger.info(f"Parameters updated: {parameters}")
    
    async def cleanup(self):
        """
        リソースのクリーンアップ
        """
        # 処理スレッドを停止
        self.audio_queue.put(None)
        self.processing_thread.join(timeout=5)
        
        logger.info("OpenAIWhisperProcessor cleaned up")
