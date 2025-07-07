import React, { useState, useRef, useEffect } from 'react';

interface Transcription {
  text: string;
  is_final: boolean;
  start: number;
  end: number;
  timestamp: number;
}

interface Stats {
  active_connections: number;
  status: string;
  whisper_model: string;
  whisper_device: string;
  gpu_vram_used_gb: number | null;
  gpu_vram_total_gb: number | null;
  gpu_vram_usage_percent: number | null;
  gpu_name: string | null;
  model_sharing: {
    loaded_models: Array<{
      backend: string;
      loaded_at: string;
      last_used: string;
      active_users: number;
      usage_count: number;
    }>;
    total_active_users: number;
    total_usage_count: number;
  } | null;
}

interface Parameters {
  prompt: string;
  beam_size: number;
  chunk_duration: number;
  temperature: number;
  vad_filter: boolean;
  vad_min_silence_duration_ms: number;
  vad_max_speech_duration_s: number;
  repetition_penalty: number;
  no_speech_threshold: number;
}

type AudioSource = 'screen' | 'microphone' | 'both';
type Language = 'auto' | 'ja' | 'en';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([]);
  const [language, setLanguage] = useState<Language>('auto');
  const [isSilent, setIsSilent] = useState(false);  // 無音状態
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [showCopiedToast, setShowCopiedToast] = useState(false);
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [summary, setSummary] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : true;
  });
  const [audioSource, setAudioSource] = useState<AudioSource>('screen');
  const [whisperBackend, setWhisperBackend] = useState<'faster-whisper' | 'openai-whisper' | 'whisperx'>('faster-whisper');
  const [showParameters, setShowParameters] = useState(false);
  const [parameters, setParameters] = useState<Parameters>({
    prompt: '',
    beam_size: 5,
    chunk_duration: 10,
    temperature: 0.0,
    vad_filter: true,
    vad_min_silence_duration_ms: 1000,
    vad_max_speech_duration_s: 30,
    repetition_penalty: 1.1,
    no_speech_threshold: 0.6
  });
  const [enableCorrection, setEnableCorrection] = useState(false);
  const [correctedTranscriptions, setCorrectedTranscriptions] = useState<Map<number, string>>(new Map());
  const [summaryText, setSummaryText] = useState('');
  const [isCorrectLoading, setIsCorrectLoading] = useState(false);
  const [showTimestamps, setShowTimestamps] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);  // 自動スクロールのON/OFF
  
  const wsRef = useRef<WebSocket | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const transcriptPanelRef = useRef<HTMLDivElement>(null);
  const correctedPanelRef = useRef<HTMLDivElement>(null);
  const silenceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const correctionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket URLを環境変数から取得（デフォルト値あり）
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/stt';
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // ダークモード切替時にlocalStorageに保存
  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  // 一定の文章ごとにLLMで誤字修正（5分ごと）
  useEffect(() => {
    if (!enableCorrection) {
      if (correctionIntervalRef.current) {
        clearInterval(correctionIntervalRef.current);
        correctionIntervalRef.current = null;
      }
      return;
    }

    // 即座に一度実行
    if (transcriptions.length > 0) {
      correctTranscriptions();
    }

    // 5分ごとに実行
    correctionIntervalRef.current = setInterval(() => {
      if (transcriptions.length > 0) {
        correctTranscriptions();
      }
    }, 5 * 60 * 1000);

    return () => {
      if (correctionIntervalRef.current) {
        clearInterval(correctionIntervalRef.current);
      }
    };
  }, [enableCorrection, transcriptions]);

  // 統計情報を定期的に取得
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${apiUrl}/stats`);
        const data = await response.json();
        console.log('Stats received:', data);
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      }
    };

    // 初回取得
    fetchStats();

    // 5秒ごとに更新
    const interval = setInterval(fetchStats, 5000);

    return () => clearInterval(interval);
  }, [apiUrl]);

  // 新しい文字起こしが追加されたら自動スクロール（autoScrollがONの場合のみ）
  useEffect(() => {
    if (autoScroll) {
      if (transcriptPanelRef.current) {
        transcriptPanelRef.current.scrollTop = transcriptPanelRef.current.scrollHeight;
      }
      if (correctedPanelRef.current) {
        correctedPanelRef.current.scrollTop = correctedPanelRef.current.scrollHeight;
      }
    }
  }, [transcriptions, autoScroll]);

  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');
    return `${hours}:${minutes}:${seconds}`;
  };

  const copyAllTranscriptions = () => {
    // すべての確定した文字起こしを時系列順に結合
    const allText = transcriptions
      .filter(t => t.is_final)
      .sort((a, b) => a.timestamp - b.timestamp)  // 時系列順にソート
      .map(t => t.text.trim())
      .join(' ');
    
    if (!allText) {
      console.log('No text to copy');
      return;
    }
    
    navigator.clipboard.writeText(allText).then(() => {
      console.log('Copied text:', allText);
      setShowCopiedToast(true);
      setTimeout(() => setShowCopiedToast(false), 2000);
    }).catch(err => {
      console.error('Failed to copy:', err);
    });
  };

  const summarizeTranscriptions = async () => {
    setIsLoading(true);
    
    const allText = transcriptions
      .filter(t => t.is_final)
      .map(t => t.text)
      .join(' ');
    
    try {
      const response = await fetch(`${apiUrl}/api/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: allText })
      });
      const data = await response.json();
      setSummaryText(data.summary);
    } catch (error) {
      console.error('Summary failed:', error);
      setSummaryText('要約の生成に失敗しました。');
    } finally {
      setIsLoading(false);
    }
  };

  const correctTranscriptions = async () => {
    if (transcriptions.length === 0) return;
    
    setIsCorrectLoading(true);
    
    // 最新の10件の文字起こしを取得して修正
    const recentTranscriptions = transcriptions
      .filter(t => t.is_final)
      .slice(-10);
    
    try {
      const response = await fetch(`${apiUrl}/api/correct`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          transcriptions: recentTranscriptions.map(t => ({
            text: t.text,
            timestamp: t.timestamp
          }))
        })
      });
      const data = await response.json();
      
      // 修正結果を保存
      const newCorrected = new Map(correctedTranscriptions);
      data.corrections.forEach((correction: any) => {
        newCorrected.set(correction.timestamp, correction.corrected_text);
      });
      setCorrectedTranscriptions(newCorrected);
    } catch (error) {
      console.error('Correction failed:', error);
    } finally {
      setIsCorrectLoading(false);
    }
  };

  const clearTranscriptions = () => {
    setTranscriptions([]);
    setCorrectedTranscriptions(new Map());
    setSummaryText('');
  };

  // 音声ストリームを取得する関数
  const getAudioStreams = async (): Promise<{ screenStream?: MediaStream; micStream?: MediaStream }> => {
    let screenStream: MediaStream | undefined;
    let micStream: MediaStream | undefined;

    try {
      // 画面音声の取得
      if (audioSource === 'screen' || audioSource === 'both') {
        const displayStream = await navigator.mediaDevices.getDisplayMedia({
          video: {
            cursor: 'never'  // カーソルを除外
          },
          audio: {
            channelCount: 2,
            echoCancellation: false,
            noiseSuppression: false,
            sampleRate: 48000,
            suppressLocalAudioPlayback: false  // ローカル音声再生を維持
          },
          preferCurrentTab: false,  // 現在のタブを優先しない（すべてのオプションを表示）
          systemAudio: 'include'  // システム音声を含める
        });
        
        // 選択したソースの情報をログに出力
        const videoTrack = displayStream.getVideoTracks()[0];
        if (videoTrack) {
          const settings = videoTrack.getSettings();
          console.log('Sharing source:', settings.displaySurface || 'unknown');
        }
        
        // ビデオトラックは不要なので削除
        const videoTracks = displayStream.getVideoTracks();
        videoTracks.forEach(track => {
          track.stop();
          displayStream.removeTrack(track);
        });
        
        // 音声トラックがあることを確認
        const audioTracks = displayStream.getAudioTracks();
        if (audioTracks.length === 0) {
          // Windowsやアプリケーション共有の場合、音声が取得できないことがある
          throw new Error('No audio track found. Please make sure to:\n1. Share a browser tab (not window or screen), OR\n2. Check "Share audio" option in the sharing dialog, OR\n3. Use microphone option for system audio capture.');
        }
        
        screenStream = displayStream;
      }

      // マイク音声の取得
      if (audioSource === 'microphone' || audioSource === 'both') {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 48000
          }
        });
      }

      return { screenStream, micStream };
    } catch (error) {
      // 既に取得したストリームをクリーンアップ
      if (screenStream) {
        screenStream.getTracks().forEach(track => track.stop());
      }
      if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
      }
      throw error;
    }
  };

  // 複数の音声ストリームをミックスする関数
  const createMixedAudioStream = (audioContext: AudioContext, streams: MediaStream[]): MediaStreamAudioSourceNode => {
    const sources = streams.map(stream => audioContext.createMediaStreamSource(stream));
    
    if (sources.length === 1) {
      return sources[0];
    }
    
    // 複数のソースをミックス
    const gainNodes = sources.map(source => {
      const gainNode = audioContext.createGain();
      gainNode.gain.value = 1.0 / sources.length; // 音量を調整
      source.connect(gainNode);
      return gainNode;
    });
    
    const mixerNode = audioContext.createGain();
    gainNodes.forEach(gainNode => gainNode.connect(mixerNode));
    
    // MediaStreamDestinationNodeを作成してミックスした音声を出力
    const destination = audioContext.createMediaStreamDestination();
    mixerNode.connect(destination);
    
    // ミックスしたストリームから新しいSourceNodeを作成
    return audioContext.createMediaStreamSource(destination.stream);
  };

  const startRecording = async () => {
    try {
      // 音声ストリームを取得
      const { screenStream, micStream } = await getAudioStreams();
      
      screenStreamRef.current = screenStream || null;
      micStreamRef.current = micStream || null;

      // WebSocket接続
      setConnectionStatus('connecting');
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      // 接続時に言語設定を送信
      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('connected');
        // 言語設定とバックエンド設定を送信
        ws.send(JSON.stringify({
          type: 'config',
          language: language === 'auto' ? null : language,
          backend: whisperBackend,
          parameters: parameters
        }));
        // 音声ソース情報を送信
        ws.send(JSON.stringify({
          type: 'audio_source',
          source: audioSource
        }));
      };

      ws.onmessage = (event) => {
        const transcription: Transcription = JSON.parse(event.data);
        console.log('Received transcription:', transcription);
        
        setTranscriptions(prev => {
          // 同じタイムスタンプの文字起こしがある場合は結合
          const existingIndex = prev.findIndex(t => 
            t.timestamp === transcription.timestamp && t.is_final === transcription.is_final
          );
          
          if (existingIndex !== -1) {
            // 同じタイムスタンプのものがある場合は結合
            const newTranscriptions = [...prev];
            const existing = newTranscriptions[existingIndex];
            
            // テキストが異なる場合のみ結合
            if (existing.text !== transcription.text && !existing.text.includes(transcription.text)) {
              newTranscriptions[existingIndex] = {
                ...existing,
                text: existing.text + ' ' + transcription.text,
                end: Math.max(existing.end, transcription.end)
              };
              console.log('Merged transcription:', newTranscriptions[existingIndex]);
            }
            return newTranscriptions;
          }
          
          // 重複チェック（同じテキストが既に存在する場合はスキップ）
          const isDuplicate = prev.some(t => 
            t.text === transcription.text && 
            Math.abs(t.timestamp - transcription.timestamp) < 5
          );
          
          if (isDuplicate) {
            console.log('Skipping duplicate transcription:', transcription.text);
            return prev;
          }
          
          // 類似テキストチェック（編集距離が近い場合はスキップ）
          const isSimilar = prev.some(t => {
            if (t.text.length > 10 && transcription.text.length > 10) {
              const similarity = calculateSimilarity(t.text, transcription.text);
              return similarity > 0.85;
            }
            return false;
          });
          
          if (isSimilar) {
            console.log('Skipping similar transcription:', transcription.text);
            return prev;
          }
          
          const filtered = prev.filter(t => t.is_final || t.timestamp !== transcription.timestamp);
          return [...filtered, transcription];
        });
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setConnectionStatus('disconnected');
      };

      // Web Audio APIで音声を処理
      const audioContext = new AudioContext({ sampleRate: 48000 });
      audioContextRef.current = audioContext;
      
      // 利用可能なストリームを収集
      const availableStreams: MediaStream[] = [];
      if (screenStream) availableStreams.push(screenStream);
      if (micStream) availableStreams.push(micStream);
      
      // 音声ソースを作成（単一または混合）
      const source = createMixedAudioStream(audioContext, availableStreams);
      
      const bufferSize = 4096;
      const processor = audioContext.createScriptProcessor(bufferSize, source.channelCount || 1, 1);
      processorRef.current = processor;
      
      let pcmBuffer: Float32Array[] = [];
      let totalSamples = 0;
      const targetSampleRate = 16000;
      const sendInterval = 1000;
      
      processor.onaudioprocess = (e) => {
        let monoData: Float32Array;
        
        if (e.inputBuffer.numberOfChannels > 1) {
          const leftChannel = e.inputBuffer.getChannelData(0);
          const rightChannel = e.inputBuffer.getChannelData(1);
          monoData = new Float32Array(leftChannel.length);
          for (let i = 0; i < leftChannel.length; i++) {
            monoData[i] = (leftChannel[i] + rightChannel[i]) / 2;
          }
        } else {
          monoData = new Float32Array(e.inputBuffer.getChannelData(0));
        }
        
        pcmBuffer.push(monoData);
        totalSamples += monoData.length;
      };
      
      const intervalId = setInterval(() => {
        if (pcmBuffer.length > 0 && ws.readyState === WebSocket.OPEN) {
          const combinedBuffer = new Float32Array(totalSamples);
          let offset = 0;
          for (const buffer of pcmBuffer) {
            combinedBuffer.set(buffer, offset);
            offset += buffer.length;
          }
          
          const downsampleRatio = audioContext.sampleRate / targetSampleRate;
          const downsampledLength = Math.floor(combinedBuffer.length / downsampleRatio);
          const downsampledBuffer = new Float32Array(downsampledLength);
          
          for (let i = 0; i < downsampledLength; i++) {
            const startIdx = Math.floor(i * downsampleRatio);
            const endIdx = Math.floor((i + 1) * downsampleRatio);
            let sum = 0;
            let count = 0;
            for (let j = startIdx; j < endIdx && j < combinedBuffer.length; j++) {
              sum += combinedBuffer[j];
              count++;
            }
            downsampledBuffer[i] = count > 0 ? sum / count : 0;
          }
          
          const int16Buffer = new Int16Array(downsampledLength);
          let maxAmplitude = 0;
          for (let i = 0; i < downsampledLength; i++) {
            const s = Math.max(-1, Math.min(1, downsampledBuffer[i]));
            int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            maxAmplitude = Math.max(maxAmplitude, Math.abs(s));
          }
          
          console.log(`Sending PCM data: ${int16Buffer.byteLength} bytes, max amplitude: ${maxAmplitude.toFixed(3)}, source: ${audioSource}`);
          
          if (maxAmplitude > 0.001) {
            // バイナリデータを直接送信
            ws.send(int16Buffer.buffer);
            setIsSilent(false);
            
            // 無音タイマーをリセット
            if (silenceTimeoutRef.current) {
              clearTimeout(silenceTimeoutRef.current);
            }
          } else {
            console.log('Skipping silent audio chunk');
            setIsSilent(true);
            
            // 3秒の無音後に自動的にバッファをクリア
            if (!silenceTimeoutRef.current) {
              silenceTimeoutRef.current = setTimeout(() => {
                console.log('Long silence detected, clearing buffer');
                silenceTimeoutRef.current = null;
              }, 3000);
            }
          }
          
          pcmBuffer = [];
          totalSamples = 0;
        }
      }, sendInterval);
      
      // @ts-ignore
      processor.intervalId = intervalId;
      
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      setIsRecording(true);
      console.log(`Audio processing started with source: ${audioSource}`);

    } catch (error) {
      console.error('Error starting recording:', error);
      
      let errorMessage = 'Failed to start recording. ';
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          errorMessage += 'Permission denied. Please allow microphone/screen sharing access.';
        } else if (error.name === 'NotFoundError') {
          errorMessage += 'No audio source available. Please check your microphone or screen sharing settings.';
        } else if (error.name === 'NotReadableError') {
          errorMessage += 'Could not access the selected audio source.';
        } else {
          errorMessage += error.message;
        }
      }
      alert(errorMessage);
    }
  };

  // 文字列の類似度を計算（レーベンシュタイン距離ベース）
  const calculateSimilarity = (str1: string, str2: string): number => {
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;
    
    if (longer.length === 0) {
      return 1.0;
    }
    
    const editDistance = (longer: string, shorter: string): number => {
      const matrix: number[][] = [];
      
      for (let i = 0; i <= shorter.length; i++) {
        matrix[i] = [i];
      }
      
      for (let j = 0; j <= longer.length; j++) {
        matrix[0][j] = j;
      }
      
      for (let i = 1; i <= shorter.length; i++) {
        for (let j = 1; j <= longer.length; j++) {
          if (shorter.charAt(i - 1) === longer.charAt(j - 1)) {
            matrix[i][j] = matrix[i - 1][j - 1];
          } else {
            matrix[i][j] = Math.min(
              matrix[i - 1][j - 1] + 1,
              matrix[i][j - 1] + 1,
              matrix[i - 1][j] + 1
            );
          }
        }
      }
      
      return matrix[shorter.length][longer.length];
    };
    
    const distance = editDistance(longer.toLowerCase(), shorter.toLowerCase());
    return (longer.length - distance) / longer.length;
  };
  
  const stopRecording = () => {
    if (processorRef.current) {
      // @ts-ignore
      if (processorRef.current.intervalId) {
        // @ts-ignore
        clearInterval(processorRef.current.intervalId);
      }
      processorRef.current.disconnect();
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    if (screenStreamRef.current) {
      screenStreamRef.current.getTracks().forEach(track => track.stop());
    }

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
    }

    setIsRecording(false);
    setConnectionStatus('disconnected');
  };

  const getAudioSourceDisplayName = (source: AudioSource) => {
    switch (source) {
      case 'screen': return 'Screen/Tab Audio';
      case 'microphone': return 'Microphone Only';
      case 'both': return 'Screen + Microphone';
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      darkMode ? 'bg-gray-900' : 'bg-gray-100'
    }`}>
      <div className="h-screen flex flex-col">
        {/* ヘッダー */}
        <div className={`flex-shrink-0 shadow-md p-4 ${
          darkMode ? 'bg-gray-800' : 'bg-white'
        }`}>
          <div className="max-w-7xl mx-auto">
            <div className="flex items-center justify-between mb-3">
              <h1 className={`text-2xl font-bold ${
                darkMode ? 'text-gray-100' : 'text-gray-800'
              }`}>
                Whisper Real-time Transcription
              </h1>
              
              {/* ダークモード切替 */}
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-2 rounded-lg transition-colors ${
                  darkMode 
                    ? 'bg-gray-700 hover:bg-gray-600 text-yellow-400' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                }`}
                title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              >
                {darkMode ? '🌞' : '🌙'}
              </button>
            </div>
            
            {/* 音声ソース選択、言語選択、バックエンド選択 */}
            {!isRecording && (
              <div className="space-y-3 mb-4">
                {/* 音声ソースと言語を同じ行に */}
                <div className="flex gap-4">
                  {/* 音声ソース選択 */}
                  <div className="flex-1">
                    <label className={`text-xs font-medium mb-1 block ${
                      darkMode ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      Audio Source
                    </label>
                    <select
                      value={audioSource}
                      onChange={(e) => setAudioSource(e.target.value as AudioSource)}
                      className={`w-full px-3 py-2 rounded-md text-sm ${
                        darkMode
                          ? 'bg-gray-700 text-gray-200 border-gray-600'
                          : 'bg-white text-gray-800 border-gray-300'
                      } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    >
                      <option value="screen">🖥️ Screen/Tab</option>
                      <option value="microphone">🎤 Microphone</option>
                      <option value="both">🖥️+🎤 Both</option>
                    </select>
                  </div>
                  
                  {/* 言語選択 */}
                  <div className="flex-1">
                    <label className={`text-xs font-medium mb-1 block ${
                      darkMode ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      Language
                    </label>
                    <select
                      value={language}
                      onChange={(e) => setLanguage(e.target.value as Language)}
                      className={`w-full px-3 py-2 rounded-md text-sm ${
                        darkMode
                          ? 'bg-gray-700 text-gray-200 border-gray-600'
                          : 'bg-white text-gray-800 border-gray-300'
                      } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    >
                      <option value="auto">Auto Detect</option>
                      <option value="ja">日本語</option>
                      <option value="en">English</option>
                    </select>
                  </div>
                  
                  {/* バックエンド選択 */}
                  <div className="flex-1">
                    <label className={`text-xs font-medium mb-1 block ${
                      darkMode ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      Engine
                    </label>
                    <select
                      value={whisperBackend}
                      onChange={(e) => setWhisperBackend(e.target.value as 'faster-whisper' | 'openai-whisper' | 'whisperx')}
                      className={`w-full px-3 py-2 rounded-md text-sm ${
                        darkMode
                          ? 'bg-gray-700 text-gray-200 border-gray-600'
                          : 'bg-white text-gray-800 border-gray-300'
                      } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    >
                      <option value="faster-whisper">Faster Whisper</option>
                      {/* 一時的に無効化 - コメントを外すと有効化されます */}
                      {/* <option value="openai-whisper">OpenAI Whisper</option> */}
                      {/* <option value="whisperx">WhisperX 3.4.2</option> */}
                      <option value="openai-whisper" disabled className="text-gray-400">OpenAI Whisper (準備中)</option>
                      <option value="whisperx" disabled className="text-gray-400">WhisperX 3.4.2 (準備中)</option>
                    </select>
                  </div>
                </div>
              </div>
            )}
            
            {/* ステータスとコントロール */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                {/* 接続ステータス */}
                <div className="flex items-center gap-2">
                  <span className="text-xl">
                    {connectionStatus === 'connected' ? '🟢' : '🔴'}
                  </span>
                  <span className={`font-semibold ${
                    darkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    {connectionStatus === 'connected' ? 'Live' : 'Stopped'}
                  </span>
                  {isRecording && (
                    <>
                      <span className={`text-sm ${
                        darkMode ? 'text-gray-400' : 'text-gray-600'
                      }`}>
                        ({getAudioSourceDisplayName(audioSource)})
                      </span>
                      {isSilent && (
                        <span className={`ml-2 text-sm ${
                          darkMode ? 'text-yellow-400' : 'text-yellow-600'
                        }`}>
                          🔇 Silent
                        </span>
                      )}
                    </>
                  )}
                </div>
                
                {/* 同時接続数とGPU情報 */}
                {stats && (
                  <>
                    <div className={`text-sm ${
                      darkMode ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      <span>👥 Active connections: {stats.active_connections}</span>
                    </div>
                    {stats.gpu_vram_used_gb !== null && stats.gpu_vram_total_gb !== null && (
                      <div className="text-sm flex items-center gap-2">
                        <span>🎮</span>
                        <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                          {stats.gpu_name}:
                        </span>
                        <span className={`font-medium ${
                          stats.gpu_vram_usage_percent! > 90 ? 'text-red-500' :
                          stats.gpu_vram_usage_percent! > 70 ? 'text-yellow-500' :
                          'text-green-500'
                        }`}>
                          {stats.gpu_vram_used_gb}GB / {stats.gpu_vram_total_gb}GB
                        </span>
                        <div className="relative w-24 h-4 bg-gray-300 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className={`absolute left-0 top-0 h-full transition-all duration-500 ${
                              stats.gpu_vram_usage_percent! > 90 ? 'bg-red-500' :
                              stats.gpu_vram_usage_percent! > 70 ? 'bg-yellow-500' :
                              'bg-green-500'
                            }`}
                            style={{ width: `${stats.gpu_vram_usage_percent}%` }}
                          />
                        </div>
                        <span className={`text-xs font-medium ${
                          stats.gpu_vram_usage_percent! > 90 ? 'text-red-500' :
                          stats.gpu_vram_usage_percent! > 70 ? 'text-yellow-500' :
                          'text-green-500'
                        }`}>
                          {stats.gpu_vram_usage_percent}%
                        </span>
                      </div>
                    )}
                    {stats.model_sharing && stats.model_sharing.loaded_models.length > 0 && (
                      <div className="text-sm flex items-center gap-2">
                        <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                          📀 Models: {stats.model_sharing.loaded_models.map(m => 
                            `${m.backend} (${m.active_users} users)`
                          ).join(', ')}
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* コントロールボタン */}
              <div className="flex gap-2">
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`px-6 py-2 rounded-lg font-semibold text-white transition-colors ${
                    isRecording 
                      ? 'bg-red-500 hover:bg-red-600' 
                      : 'bg-blue-500 hover:bg-blue-600'
                  }`}
                >
                  {isRecording ? 'Stop Caption' : 'Start Caption'}
                </button>
                
                <button
                  onClick={() => setShowParameters(!showParameters)}
                  className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                    darkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                      : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                  }`}
                >
                  ⚙️ 詳細パラメータ
                </button>
                
                <button
                  onClick={clearTranscriptions}
                  className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                    darkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                      : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                  }`}
                  disabled={transcriptions.length === 0}
                >
                  🗑️ クリア
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* 詳細パラメータパネル */}
        {showParameters && (
          <div className={`flex-shrink-0 border-t ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'} p-4`}>
            <div className="max-w-7xl mx-auto">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <label className={`text-xs font-medium mb-1 block ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Prompt
                  </label>
                  <input
                    type="text"
                    value={parameters.prompt}
                    onChange={(e) => setParameters({...parameters, prompt: e.target.value})}
                    className={`w-full px-3 py-2 rounded-md text-sm ${
                      darkMode
                        ? 'bg-gray-700 text-gray-200 border-gray-600'
                        : 'bg-white text-gray-800 border-gray-300'
                    } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    placeholder="Optional prompt..."
                  />
                </div>
                
                <div>
                  <label className={`text-xs font-medium mb-1 block ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Beam Size
                  </label>
                  <input
                    type="number"
                    value={parameters.beam_size}
                    onChange={(e) => setParameters({...parameters, beam_size: parseInt(e.target.value) || 5})}
                    className={`w-full px-3 py-2 rounded-md text-sm ${
                      darkMode
                        ? 'bg-gray-700 text-gray-200 border-gray-600'
                        : 'bg-white text-gray-800 border-gray-300'
                    } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    min="1"
                    max="10"
                  />
                </div>
                
                <div>
                  <label className={`text-xs font-medium mb-1 block ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Chunk Duration (s)
                  </label>
                  <input
                    type="number"
                    value={parameters.chunk_duration}
                    onChange={(e) => setParameters({...parameters, chunk_duration: parseInt(e.target.value) || 10})}
                    className={`w-full px-3 py-2 rounded-md text-sm ${
                      darkMode
                        ? 'bg-gray-700 text-gray-200 border-gray-600'
                        : 'bg-white text-gray-800 border-gray-300'
                    } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    min="5"
                    max="30"
                  />
                </div>
                
                <div>
                  <label className={`text-xs font-medium mb-1 block ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Temperature
                  </label>
                  <input
                    type="number"
                    value={parameters.temperature}
                    onChange={(e) => setParameters({...parameters, temperature: parseFloat(e.target.value) || 0})}
                    className={`w-full px-3 py-2 rounded-md text-sm ${
                      darkMode
                        ? 'bg-gray-700 text-gray-200 border-gray-600'
                        : 'bg-white text-gray-800 border-gray-300'
                    } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    min="0"
                    max="1"
                    step="0.1"
                  />
                </div>
                
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="vad-filter"
                    checked={parameters.vad_filter}
                    onChange={(e) => setParameters({...parameters, vad_filter: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <label htmlFor="vad-filter" className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    VAD Filter
                  </label>
                </div>
                
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="show-timestamps"
                    checked={showTimestamps}
                    onChange={(e) => setShowTimestamps(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <label htmlFor="show-timestamps" className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    タイムスタンプ表示
                  </label>
                </div>
                
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="enable-correction"
                    checked={enableCorrection}
                    onChange={(e) => setEnableCorrection(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <label htmlFor="enable-correction" className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    LLM誤字修正
                  </label>
                  {isCorrectLoading && (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  )}
                </div>
                
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="auto-scroll"
                    checked={autoScroll}
                    onChange={(e) => setAutoScroll(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <label htmlFor="auto-scroll" className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    自動スクロール
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* メインコンテンツエリア */}
        <div className="flex-grow flex flex-col p-4 min-h-0">
          <div className="w-full max-w-7xl mx-auto h-full flex flex-col gap-4">
            {/* 字幕表示エリア */}
            <div className="flex-grow flex gap-4 min-h-0">
              {/* 文字起こし表示 */}
              <div className={`${enableCorrection ? 'w-1/2' : 'w-full'} bg-black bg-opacity-75 rounded-lg shadow-2xl flex flex-col min-h-0`}>
                {/* パネルヘッダー */}
                <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
                  <h2 className="text-white font-semibold text-lg">Live Transcription</h2>
                  <div className="flex gap-2">
                    <button
                      onClick={copyAllTranscriptions}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors"
                      disabled={transcriptions.filter(t => t.is_final).length === 0}
                    >
                      Copy
                    </button>
                    <button
                      onClick={summarizeTranscriptions}
                      className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md text-sm font-medium transition-colors"
                      disabled={transcriptions.filter(t => t.is_final).length === 0}
                    >
                      Summarize
                    </button>
                  </div>
                </div>
                
                {/* 文字起こし表示エリア */}
                <div 
                  ref={transcriptPanelRef}
                  className="flex-grow p-6 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800 min-h-0"
                  style={{ overscrollBehavior: 'contain' }}
                >
                  {transcriptions.length === 0 && isRecording && (
                    <p className="text-white text-center opacity-50 text-lg">
                      Waiting for speech...
                    </p>
                  )}
                  
                  {transcriptions.map((trans, index) => {
                    return (
                      <div
                        key={`${trans.timestamp}-${index}`}
                        className={`text-white mb-2 ${
                          trans.is_final ? 'opacity-100' : 'opacity-70'
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          {showTimestamps && (
                            <span className="text-xs text-gray-400 mt-1">
                              [{formatTime(trans.timestamp)}]
                            </span>
                          )}
                          <span className="text-base">{trans.text}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              
              {/* 修正された文字起こし表示 */}
              {enableCorrection && (
                <div className="w-1/2 bg-black bg-opacity-75 rounded-lg shadow-2xl flex flex-col min-h-0">
                  <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
                    <h2 className="text-white font-semibold text-lg">LLM Corrected</h2>
                  </div>
                  
                  <div 
                    ref={correctedPanelRef}
                    className="flex-grow p-6 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800 min-h-0"
                    style={{ overscrollBehavior: 'contain' }}
                  >
                    {transcriptions.filter(t => t.is_final).map((trans, index) => {
                      const corrected = correctedTranscriptions.get(trans.timestamp);
                      return (
                        <div
                          key={`corrected-${trans.timestamp}-${index}`}
                          className="text-white mb-2"
                        >
                          <div className="flex items-start gap-2">
                            {showTimestamps && (
                              <span className="text-xs text-gray-400 mt-1">
                                [{formatTime(trans.timestamp)}]
                              </span>
                            )}
                            <span className="text-base">
                              {corrected || trans.text}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
            
            {/* 要約表示枠 */}
            {summaryText && (
              <div className="h-48 bg-black bg-opacity-75 rounded-lg shadow-2xl flex flex-col flex-shrink-0">
                <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
                  <h2 className="text-white font-semibold text-lg">Summary</h2>
                  <button
                    onClick={() => setSummaryText('')}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    ✕
                  </button>
                </div>
                
                <div className="flex-grow p-6 overflow-y-auto">
                  <div className="text-white whitespace-pre-wrap">{summaryText}</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Copiedトースト */}
        {showCopiedToast && (
          <div className="fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg animate-fade-in-out">
            Copied!
          </div>
        )}

        {/* サマリーモーダル */}
        {showSummaryModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className={`rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto ${
              darkMode ? 'bg-gray-800' : 'bg-white'
            }`}>
              <h3 className={`text-xl font-bold mb-4 ${
                darkMode ? 'text-gray-100' : 'text-gray-800'
              }`}>
                Summary
              </h3>
              
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              ) : (
                <div className={`mb-6 whitespace-pre-wrap ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  {summary}
                </div>
              )}
              
              <button
                onClick={() => {
                  setShowSummaryModal(false);
                  if (summary) {
                    setSummaryText(summary);
                  }
                }}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium transition-colors"
                disabled={isLoading}
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Tailwindのアニメーション用スタイル */}
      <style jsx>{`
        @keyframes fade-in-out {
          0% { opacity: 0; transform: translateY(-10px); }
          20% { opacity: 1; transform: translateY(0); }
          80% { opacity: 1; transform: translateY(0); }
          100% { opacity: 0; transform: translateY(-10px); }
        }
        
        .animate-fade-in-out {
          animation: fade-in-out 2s ease-in-out;
        }
        
        /* スクロールバーのカスタマイズ */
        .scrollbar-thin::-webkit-scrollbar {
          width: 8px;
        }
        
        .scrollbar-thin::-webkit-scrollbar-track {
          background: #1f2937;
          border-radius: 4px;
        }
        
        .scrollbar-thin::-webkit-scrollbar-thumb {
          background: #4b5563;
          border-radius: 4px;
        }
        
        .scrollbar-thin::-webkit-scrollbar-thumb:hover {
          background: #6b7280;
        }
      `}</style>
    </div>
  );
}

export default App;