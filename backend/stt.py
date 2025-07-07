"""
Speech-to-Text processing module using faster-whisper
faster-whisperを使用した音声文字起こし処理モジュール
"""

import asyncio
import subprocess
import numpy as np
import logging
import time
from typing import List, Dict, Optional, AsyncGenerator
from collections import deque
import os
import threading
import queue
from faster_whisper import WhisperModel
import struct
from difflib import SequenceMatcher
from model_manager import model_manager

logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    """
    音声データを処理し、Whisperで文字起こしを行うクラス
    """
    
    def __init__(self):
        """
        環境変数から設定を読み込んで初期化
        """
        self.language = None  # None = 自動検出
        self.model = None  # モデルは必要時に取得
        self.backend_type = 'faster-whisper'
        
        # 環境変数から設定を読み込み
        self.model_size = os.getenv('WHISPER_MODEL', 'large-v3')
        self.device = os.getenv('WHISPER_DEVICE', 'cuda')
        self.compute_type = os.getenv('WHISPER_COMPUTE_TYPE', 'float16')
        
        # Whisperパラメータ（最適化された設定）
        self.beam_size = int(os.getenv('WHISPER_BEAM_SIZE', '5'))  # 適度なビームサイズ
        self.best_of = int(os.getenv('WHISPER_BEST_OF', '5'))  # 適度な候補数
        self.patience = float(os.getenv('WHISPER_PATIENCE', '1.0'))  # 標準的な探索時間
        self.temperature = float(os.getenv('WHISPER_TEMPERATURE', '0.0'))  # 確定的な結果
        self.no_speech_threshold = float(os.getenv('WHISPER_NO_SPEECH_THRESHOLD', '0.6'))
        self.repetition_penalty = float(os.getenv('WHISPER_REPETITION_PENALTY', '1.1'))  # 軽い繰り返し抑制
        self.compression_ratio_threshold = float(os.getenv('WHISPER_COMPRESSION_RATIO_THRESHOLD', '2.4'))
        self.length_penalty = float(os.getenv('WHISPER_LENGTH_PENALTY', '1.0'))
        self.no_repeat_ngram_size = int(os.getenv('WHISPER_NO_REPEAT_NGRAM_SIZE', '0'))
        self.log_prob_threshold = float(os.getenv('WHISPER_LOG_PROB_THRESHOLD', '-1.0'))
        self.condition_on_previous_text = False  # 前の文脈を使用しない（重複の原因）
        self.prompt = os.getenv('WHISPER_PROMPT', None) or None
        self.prefix = os.getenv('WHISPER_PREFIX', None) or None
        self.suppress_blank = os.getenv('WHISPER_SUPPRESS_BLANK', 'true').lower() == 'true'
        self.suppress_tokens = os.getenv('WHISPER_SUPPRESS_TOKENS', '-1')
        self.without_timestamps = os.getenv('WHISPER_WITHOUT_TIMESTAMPS', 'false').lower() == 'true'
        self.max_initial_timestamp = float(os.getenv('WHISPER_MAX_INITIAL_TIMESTAMP', '1.0'))
        self.word_timestamps = os.getenv('WHISPER_WORD_TIMESTAMPS', 'false').lower() == 'true'
        # Punctuations - デフォルト値をコード内で設定
        self.prepend_punctuations = '"\'¿([{-'
        self.append_punctuations = '"\'.。,，!！?？:：”)]}、'
        
        # VADパラメータ (faster-whisper 1.1.1の改善されたVADを活用)
        self.vad_filter = True
        self.vad_min_speech_duration_ms = int(os.getenv('WHISPER_VAD_MIN_SPEECH_DURATION_MS', '500'))  # より長い最小発話時間
        self.vad_max_speech_duration_s = float(os.getenv('WHISPER_VAD_MAX_SPEECH_DURATION_S', '30.0'))
        self.vad_min_silence_duration_ms = int(os.getenv('WHISPER_VAD_MIN_SILENCE_DURATION_MS', '2000'))  # より長い無音判定
        self.vad_speech_pad_ms = int(os.getenv('WHISPER_VAD_SPEECH_PAD_MS', '500'))  # より長いパディング
        
        # faster-whisper 1.1.1の新機能
        self.vad_threshold = float(os.getenv('WHISPER_VAD_THRESHOLD', '0.5'))  # VADの闾値
        self.use_silero_vad = os.getenv('WHISPER_USE_SILERO_VAD', 'true').lower() == 'true'  # Silero VADを使用
        
        # モデルの初期化は廃止（必要時に取得）
        logger.info(f"TranscriptionProcessor initialized (model will be loaded on demand)")
        
        # 音声バッファ（15秒分のPCMデータを保持）
        self.audio_buffer = deque(maxlen=15 * 16000)  # 15秒 * 16kHz
        
        # 処理済み音声のタイムスタンプを記録
        self.processed_timestamps = deque(maxlen=20)
        self.min_segment_gap = 1.0  # 最小セグメント間隔（秒）
        
        # 重複防止用の文字起こし履歴
        self.transcription_history = deque(maxlen=10)  # 最近の10件を保持
        self.last_transcription = ""
        self.last_transcription_time = 0
        
        # 無音検出用
        self.silence_duration = 0
        self.silence_threshold = 0.01
        self.silence_clear_duration = 3.0  # 3秒の無音でバッファクリア
        self.min_audio_length = 1.0  # 最小音声長（1秒）
        
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
        self.last_audio_time = time.time()  # 最後に音声を受信した時刻
        
        logger.info("TranscriptionProcessor initialized with optimized settings")
    
    async def _get_model(self):
        """
        モデルマネージャーからモデルを取得
        """
        if self.model is None:
            self.model = await model_manager.get_model(self.backend_type)
        return self.model
    
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
                logger.info(f"Received PCM chunk #{chunk_count}: {len(audio_data)} bytes, total: {total_bytes_received} bytes")
                
                # PCMデータ（Int16）をFloat32に変換
                try:
                    # データ長のチェック
                    if len(audio_data) % 2 != 0:
                        logger.warning(f"Received odd-length audio data: {len(audio_data)} bytes, trimming last byte")
                        audio_data = audio_data[:-1]
                    
                    # バイト列をInt16配列に変換
                    pcm_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    # Float32に正規化
                    audio_array = pcm_int16.astype(np.float32) / 32768.0
                    
                    logger.info(f"Converted PCM data: {len(audio_data)} bytes -> {len(audio_array)} samples")
                    
                    # 無音検出
                    max_amplitude = np.max(np.abs(audio_array))
                    current_time = time.time()
                    
                    if max_amplitude < self.silence_threshold:
                        # 無音の場合
                        self.silence_duration += current_time - self.last_audio_time
                        if self.silence_duration > self.silence_clear_duration:
                            # 長い無音でバッファをクリア
                            logger.info(f"Clearing buffer after {self.silence_duration:.1f}s of silence")
                            self.audio_buffer.clear()
                            self.transcription_history.clear()
                            self.silence_duration = 0
                    else:
                        # 音声がある場合
                        self.silence_duration = 0
                        # バッファに追加
                        self.audio_buffer.extend(audio_array)
                        logger.info(f"Audio buffer size: {len(self.audio_buffer)} samples, max amplitude: {max_amplitude}")
                    
                    self.last_audio_time = current_time
                    
                    # 定期的に文字起こしを実行（10秒チャンクで処理）
                    buffer_duration = len(self.audio_buffer) / 16000.0
                    
                    # 10秒以上のデータが溜まったら処理、または無音が続いた場合
                    should_process = False
                    if buffer_duration >= self.chunk_duration:
                        should_process = True
                        logger.info(f"Processing full chunk: {buffer_duration:.1f}s")
                    elif max_amplitude < self.silence_threshold and buffer_duration >= self.min_audio_length:
                        # 無音で最小長以上ある場合
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
    
    def _is_duplicate_or_similar(self, text: str, threshold: float = 0.8) -> bool:
        """
        テキストが重複または類似しているかチェック
        
        Args:
            text: チェックするテキスト
            threshold: 類似度の閾値（0-1）
            
        Returns:
            bool: 重複または類似している場合True
        """
        # 空のテキストはスキップ
        if not text.strip():
            return True
        
        # 履歴と比較
        for hist_text in self.transcription_history:
            # 完全一致
            if text == hist_text:
                logger.info(f"Exact duplicate found: {text}")
                return True
            
            # 類似度チェック（編集距離ベース）
            similarity = SequenceMatcher(None, text.lower(), hist_text.lower()).ratio()
            if similarity > threshold:
                logger.info(f"Similar text found (similarity: {similarity:.2f}): {text} ~ {hist_text}")
                return True
            
            # 部分文字列チェック（短い方が長い方に含まれている）
            if len(text) > 10 and len(hist_text) > 10:
                if text in hist_text or hist_text in text:
                    logger.info(f"Substring found: {text} in {hist_text}")
                    return True
        
        return False
    
    def _transcribe_buffer(self):
        """
        バッファ内の音声を文字起こし
        """
        if len(self.audio_buffer) < 16000:  # 1秒未満の場合はスキップ
            logger.warning(f"Buffer too small for transcription: {len(self.audio_buffer)} samples")
            return
        
        # モデルを取得（非同期関数を同期的に呼び出す）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            model = loop.run_until_complete(self._get_model())
        finally:
            loop.close()
        
        if model is None:
            logger.error("Failed to get model")
            return
        
        # バッファから音声データを取得（最大10秒）
        audio_data = np.array(list(self.audio_buffer))
        logger.info(f"Transcribing audio: {len(audio_data)} samples, max amplitude: {np.max(np.abs(audio_data))}")
        
        # 無音チェック
        if np.max(np.abs(audio_data)) < 0.003:  # 閾値をさらに緩和 (0.01 -> 0.003)
            logger.info("Audio appears to be silence, skipping transcription")
            return
        
        try:
            # suppress_tokensの処理
            suppress_tokens = None
            if self.suppress_tokens != '-1' and self.suppress_tokens:
                try:
                    suppress_tokens = [int(x.strip()) for x in self.suppress_tokens.split(',') if x.strip()]
                    if not suppress_tokens:
                        suppress_tokens = None
                except ValueError:
                    logger.warning(f"Invalid suppress_tokens format: {self.suppress_tokens}")
                    suppress_tokens = None
            
            # temperatureの処理 - 単一の値として渡す
            temperature = self.temperature if self.temperature > 0 else 0.0
            
            # Whisperで文字起こし（faster-whisper 1.1.1の機能を活用）
            vad_params = {
                "min_speech_duration_ms": self.vad_min_speech_duration_ms,
                "max_speech_duration_s": self.vad_max_speech_duration_s,
                "min_silence_duration_ms": self.vad_min_silence_duration_ms,
                "speech_pad_ms": self.vad_speech_pad_ms,
            }
            
            # faster-whisper 1.1.1ではVADの闾値も設定可能
            if hasattr(self, 'vad_threshold'):
                vad_params["threshold"] = self.vad_threshold
            
            segments, info = model.transcribe(
                audio_data,
                language=self.language,  # None = 自動検出, 'ja' = 日本語, 'en' = 英語
                task="transcribe",
                beam_size=self.beam_size,
                patience=self.patience,
                temperature=0.0,  # 確定的な結果
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=False,  # 重複を防ぐために常にFalse
                suppress_blank=True,
                vad_filter=self.vad_filter,
                vad_parameters=vad_params if self.vad_filter else None,
                # 句読点の正確な処理
                prepend_punctuations=self.prepend_punctuations,
                append_punctuations=self.append_punctuations
            )
            
            # infoログ
            logger.info(f"Transcription info - language: {info.language}, duration: {info.duration}")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return
        
        # セグメントを処理
        segment_count = 0
        current_time = time.time()
        base_timestamp = int(current_time * 1000)  # ミリ秒に変換
        
        # 同じタイムスタンプのセグメントを結合
        combined_segments = []
        current_combined_text = ""
        current_start = None
        current_end = None
        current_no_speech_prob = 1.0
        
        for segment in segments:
            segment_count += 1
            transcription_text = segment.text.strip()
            
            # 空のテキストはスキップ
            if not transcription_text:
                continue
            
            logger.info(f"Segment {segment_count}: text='{transcription_text}', no_speech_prob={segment.no_speech_prob}")
            
            # 重複・類似チェック
            if self._is_duplicate_or_similar(transcription_text):
                continue
            
            # セグメントを結合
            if current_start is None:
                current_start = segment.start
            current_end = segment.end
            current_no_speech_prob = min(current_no_speech_prob, segment.no_speech_prob)
            
            if current_combined_text:
                current_combined_text += " " + transcription_text
            else:
                current_combined_text = transcription_text
        
        # 結合されたテキストがある場合は送信
        if current_combined_text:
            # 履歴に追加
            self.transcription_history.append(current_combined_text)
            
            transcription = {
                "text": current_combined_text,
                "start": current_start or 0.0,
                "end": current_end or (len(audio_data) / 16000.0),
                "is_final": current_no_speech_prob < self.no_speech_threshold,
                "timestamp": base_timestamp
            }
            
            self.result_queue.put(transcription)
            logger.info(f"Transcribed (combined): {transcription['text']}")
            
            # 最後の文字起こしを更新
            self.last_transcription = current_combined_text
            self.last_transcription_time = current_time
        
        if segment_count == 0:
            logger.warning("No segments found in transcription")
        
        # バッファは処理前にクリアされているため、ここでは何もしない
        logger.info(f"Transcription completed with {segment_count} segments")
    
    async def process_audio_chunk(self, audio_data: bytes) -> List[Dict]:
        """
        音声チャンクを処理し、文字起こし結果を返す
        
        Args:
            audio_data: PCM形式の音声データ（16kHz, 16bit, mono）
            
        Returns:
            List[Dict]: 文字起こし結果のリスト
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
        
        Args:
            language: 'ja' (日本語), 'en' (英語), None (自動検出)
        """
        self.language = language
        logger.info(f"Language set to: {language or 'auto'}")
    
    def update_parameters(self, parameters: Dict):
        """
        Whisperパラメータを更新
        
        Args:
            parameters: 更新するパラメータの辞書
        """
        # promptの更新
        if 'prompt' in parameters:
            self.prompt = parameters['prompt'] or None
        
        # beam_sizeの更新
        if 'beam_size' in parameters:
            self.beam_size = int(parameters['beam_size'])
        
        # chunk_durationの更新（内部的には処理で使用）
        if 'chunk_duration' in parameters:
            self.chunk_duration = int(parameters['chunk_duration'])
        
        # temperatureの更新
        if 'temperature' in parameters:
            self.temperature = float(parameters['temperature'])
        
        # vad_filterの更新
        if 'vad_filter' in parameters:
            self.vad_filter = bool(parameters['vad_filter'])
        
        # vad_min_silence_duration_msの更新
        if 'vad_min_silence_duration_ms' in parameters:
            self.vad_min_silence_duration_ms = int(parameters['vad_min_silence_duration_ms'])
        
        # vad_max_speech_duration_sの更新
        if 'vad_max_speech_duration_s' in parameters:
            self.vad_max_speech_duration_s = float(parameters['vad_max_speech_duration_s'])
        
        # repetition_penaltyの更新
        if 'repetition_penalty' in parameters:
            self.repetition_penalty = float(parameters['repetition_penalty'])
        
        # no_speech_thresholdの更新
        if 'no_speech_threshold' in parameters:
            self.no_speech_threshold = float(parameters['no_speech_threshold'])
        
        logger.info(f"Parameters updated: {parameters}")
    
    async def cleanup(self):
        """
        リソースのクリーンアップ
        """
        # 処理スレッドを停止
        self.audio_queue.put(None)
        self.processing_thread.join(timeout=5)
        
        # モデルをリリース
        if self.model is not None:
            await model_manager.release_model(self.backend_type)
            self.model = None
        
        logger.info("TranscriptionProcessor cleaned up")
