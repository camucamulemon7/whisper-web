"""
Whisperモデルの共有管理システム
複数のWebSocket接続間でモデルインスタンスを共有し、メモリ効率を向上させる
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """モデル情報を管理するデータクラス"""
    model: Any
    backend_type: str
    loaded_at: datetime
    last_used: datetime
    usage_count: int = 0
    active_users: int = 0

class ModelManager:
    """
    Whisperモデルの共有管理を行うシングルトンクラス
    """
    _instance: Optional['ModelManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.models: Dict[str, ModelInfo] = {}
            self.model_lock = asyncio.Lock()
            self.idle_timeout = 300  # 5分間使用されないモデルは解放
            self.cleanup_task: Optional[asyncio.Task] = None
            self.initialized = True
            logger.info("ModelManager initialized")
    
    async def get_model(self, backend_type: str = 'faster-whisper') -> Any:
        """
        指定されたバックエンドのモデルを取得
        既に読み込まれている場合は共有インスタンスを返す
        """
        async with self.model_lock:
            # 既存のモデルがあるか確認
            if backend_type in self.models:
                model_info = self.models[backend_type]
                model_info.last_used = datetime.now()
                model_info.active_users += 1
                model_info.usage_count += 1
                logger.info(f"Reusing existing {backend_type} model. Active users: {model_info.active_users}")
                return model_info.model
            
            # 新しいモデルを読み込む
            logger.info(f"Loading new {backend_type} model...")
            model = await self._load_model(backend_type)
            
            if model is not None:
                self.models[backend_type] = ModelInfo(
                    model=model,
                    backend_type=backend_type,
                    loaded_at=datetime.now(),
                    last_used=datetime.now(),
                    usage_count=1,
                    active_users=1
                )
                logger.info(f"{backend_type} model loaded successfully")
                
                # クリーンアップタスクを開始
                if self.cleanup_task is None:
                    self.cleanup_task = asyncio.create_task(self._cleanup_idle_models())
            
            return model
    
    async def release_model(self, backend_type: str):
        """
        モデルの使用を終了（アクティブユーザー数を減らす）
        """
        async with self.model_lock:
            if backend_type in self.models:
                model_info = self.models[backend_type]
                model_info.active_users = max(0, model_info.active_users - 1)
                model_info.last_used = datetime.now()
                logger.info(f"Released {backend_type} model. Active users: {model_info.active_users}")
    
    async def _load_model(self, backend_type: str) -> Optional[Any]:
        """
        指定されたバックエンドのモデルを実際に読み込む
        """
        try:
            if backend_type == 'faster-whisper':
                from faster_whisper import WhisperModel
                model = WhisperModel(
                    model_size_or_path=os.getenv('WHISPER_MODEL', 'large-v3'),
                    device=os.getenv('WHISPER_DEVICE', 'cuda'),
                    compute_type=os.getenv('WHISPER_COMPUTE_TYPE', 'float16'),
                    num_workers=1,
                    cpu_threads=4,
                    download_root="/app/models"
                )
                # モデルを一度推論して初期化を完了させる
                logger.info("Warming up faster-whisper model...")
                try:
                    import numpy as np
                    # 1秒分の無音データをNumPy配列として作成
                    silence_audio = np.zeros(16000, dtype=np.float32)
                    segments, _ = model.transcribe(
                        silence_audio, 
                        beam_size=1,
                        language="en",  # 言語を指定して自動検出をスキップ
                        vad_filter=False  # VADを無効化して高速化
                    )
                    # ジェネレーターを消費
                    _ = list(segments)
                    logger.info("Warm-up completed successfully")
                except Exception as e:
                    logger.warning(f"Warm-up failed, but model is still usable: {e}")
                
                return model
                
            elif backend_type == 'openai-whisper':
                import whisper
                import torch
                device = os.getenv('WHISPER_DEVICE', 'cuda')
                model_name = os.getenv('WHISPER_MODEL', 'large-v3')
                
                logger.info(f"Loading OpenAI Whisper model: {model_name} on {device}")
                model = whisper.load_model(
                    model_name,
                    device=device,
                    download_root="/app/models"
                )
                
                # ウォームアップ
                logger.info("Warming up OpenAI Whisper model...")
                try:
                    import numpy as np
                    silence_audio = np.zeros(16000, dtype=np.float32)
                    _ = model.transcribe(silence_audio, language="en", fp16=False)
                    logger.info("Warm-up completed successfully")
                except Exception as e:
                    logger.warning(f"Warm-up failed, but model is still usable: {e}")
                
                return model
                
            elif backend_type == 'whisperx':
                import whisperx
                import torch
                device = os.getenv('WHISPER_DEVICE', 'cuda')
                compute_type = os.getenv('WHISPER_COMPUTE_TYPE', 'float16')
                model_name = os.getenv('WHISPER_MODEL', 'large-v3')
                
                logger.info(f"Loading WhisperX model: {model_name}")
                model = whisperx.load_model(
                    model_name,
                    device,
                    compute_type=compute_type,
                    download_root="/app/models"
                )
                
                # ウォームアップ
                logger.info("Warming up WhisperX model...")
                try:
                    import numpy as np
                    silence_audio = np.zeros(16000, dtype=np.float32)
                    _ = model.transcribe(silence_audio, batch_size=1)
                    logger.info("Warm-up completed successfully")
                except Exception as e:
                    logger.warning(f"Warm-up failed, but model is still usable: {e}")
                
                return model
            else:
                logger.error(f"Unknown backend type: {backend_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {backend_type} model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def _cleanup_idle_models(self):
        """
        アイドル状態のモデルを定期的にクリーンアップ
        """
        while True:
            try:
                await asyncio.sleep(60)  # 1分ごとにチェック
                
                async with self.model_lock:
                    current_time = datetime.now()
                    models_to_remove = []
                    
                    for backend_type, model_info in self.models.items():
                        # アクティブユーザーがいない & アイドルタイムアウトを超過
                        if (model_info.active_users == 0 and 
                            (current_time - model_info.last_used).total_seconds() > self.idle_timeout):
                            models_to_remove.append(backend_type)
                    
                    for backend_type in models_to_remove:
                        logger.info(f"Unloading idle {backend_type} model")
                        del self.models[backend_type]
                        # ガベージコレクションをトリガー
                        import gc
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    
                    # すべてのモデルが解放されたらクリーンアップタスクを終了
                    if not self.models:
                        self.cleanup_task = None
                        break
                        
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        現在のモデル使用状況を取得
        """
        stats = {
            "loaded_models": [],
            "total_active_users": 0,
            "total_usage_count": 0
        }
        
        for backend_type, model_info in self.models.items():
            stats["loaded_models"].append({
                "backend": backend_type,
                "loaded_at": model_info.loaded_at.isoformat(),
                "last_used": model_info.last_used.isoformat(),
                "active_users": model_info.active_users,
                "usage_count": model_info.usage_count
            })
            stats["total_active_users"] += model_info.active_users
            stats["total_usage_count"] += model_info.usage_count
        
        return stats

# グローバルインスタンスを作成
model_manager = ModelManager()
