import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type } from '@google/genai';
import { Video, Mic, MicOff, Play, Square, AlertCircle, Volume2, Sparkles, Eye, Settings, VolumeX, RefreshCw, Camera, FlipHorizontal, Lightbulb, Key, X, MessageCircleQuestion, ArrowRight, ScanEye, Target, UserRoundPen, Check, ChevronRight, Gauge, Save, AudioLines, Wifi, WifiOff, FileText, Loader2, BookOpen, Sun, Moon, Database, Upload, Trash2 } from 'lucide-react';
import { ConnectionState, ChatMessage, SavedSession, UserProfile, SessionSummary, ExamRecord, VariantQuestion } from '../types';
import { motion, AnimatePresence } from 'motion/react';
import { createPcmBlob, decode, decodeAudioData, blobToBase64 } from '../utils/audioUtils';
import { classifyQuestion } from '../utils/QuestionClassifier';
import { performHighPrecisionOcr } from '../utils/highPrecisionOcr';
import AudioVisualizer from './AudioVisualizer';
import Transcript from './Transcript';

import * as pdfjsLib from 'pdfjs-dist';
import Tesseract from 'tesseract.js';
// @ts-ignore
import pdfWorker from 'pdfjs-dist/build/pdf.worker.mjs?url';

// Set worker source for PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

// Global Tesseract Worker Caching to avoid re-downloading/re-initializing models constantly
let globalOcrWorker: Tesseract.Worker | null = null;
let isInitializingOcr = false;

const getOcrWorker = async (): Promise<Tesseract.Worker | null> => {
    if (globalOcrWorker) return globalOcrWorker;
    if (isInitializingOcr) {
        let retries = 0;
        while (isInitializingOcr && retries < 40) { // Wait up to 20 seconds (40 * 500ms)
            await new Promise(r => setTimeout(r, 500));
            retries++;
        }
        return globalOcrWorker;
    }
    isInitializingOcr = true;
    try {
        globalOcrWorker = await Tesseract.createWorker('chi_sim+eng');
        console.log("Global OCR Worker Initialized successfully.");
    } catch (e) {
        console.error("Failed to initialize global OCR worker", e);
    } finally {
        isInitializingOcr = false;
    }
    return globalOcrWorker;
};

// Extend Navigator interface for Network Information API (experimental)
interface NetworkInformation extends EventTarget {
  readonly downlink: number;
  readonly effectiveType: 'slow-2g' | '2g' | '3g' | '4g';
  readonly rtt: number;
  readonly saveData: boolean;
  onchange: EventListener;
}

// Configuration constants
const MODEL_NAME = 'gemini-3.1-flash-live-preview';

// Voice Options for User Selection
const VOICE_OPTIONS = [
  { id: 'Kore', name: '温柔老师 (Kore)', desc: '舒缓、平和的女性声音', gender: 'Female' },
  { id: 'Zephyr', name: '知性姐姐 (Zephyr)', desc: '清晰、专业的女性声音', gender: 'Female' },
  { id: 'Fenrir', name: '阳光哥哥 (Fenrir)', desc: '充满活力、热情的男性声音', gender: 'Male' },
  { id: 'Charon', name: '沉稳大叔 (Charon)', desc: '低沉、有磁性的男性声音', gender: 'Male' },
  { id: 'Puck', name: '幽默伙伴 (Puck)', desc: '轻松、略带调皮的男性声音', gender: 'Male' },
];



// Base instruction without user context
const BASE_SYSTEM_INSTRUCTION = `
// 核心身份与愿景
你是一个集成在智能硬件中的“苏格拉底式”启发导师。你的目标不是直接提供答案，而是通过实时视频观察学生的作业，引导其独立思考，培养学习自主性。

// 绝对行为红线（禁止事项）
1. 严禁直接给答案：无论学生如何请求，绝对禁止输出选择题选项、填空题词汇或大题的完整解题结果。
2. 禁止非学术讨论：严禁回答政治、宗教、暴力或任何违反中国法律合规要求的内容。
3. 视觉反馈优先：当观察到高拍仪画面中的题目时，优先描述你看到的关键条件，而非直接讲解。

// 教学逻辑流（必须执行）
1. 拆解（Observe）：通过视频流观察题目，先询问学生：“我看到这道题有一个关键条件，你发现了吗？”
2. 提示（Scaffolding）：若学生困惑，提供公式提示或知识点线索，而非解题步骤。
3. 出图（Visualize）：对于几何、物理或需要直观位置理解的题目，或者当学生明确说“帮我画个...”或“画个...”时，你必须调用函数工具【generateDiagram】(传入该图所表达题目的完整或关键描述字段) 来在界面生成教学示意图。生成的图像需适配显示器，线条清晰。
4. 费曼测试（Feynman Technique）：在讲解完一个知识点后，或学生表示“听懂了”之后，主动询问学生要不要做道类似的变式题。若学生同意，或当你主动发起时，你必须调用函数工具【triggerVariantQuestion】在界面生成美观的互动式变式特训，让学生在屏幕上完成选择并看你的详细步骤解析以巩固记忆。

// UI/输出规范
1. 适配显示器：输出文字需分段明确，使用大号 Markdown 标题，确保在 3 米外的电视前清晰可见。
2. 情感反馈：使用鼓励性语言（如“太棒了”、“很有创意的想法”），但避免使用复杂的渐变色或容易导致 4K 电视光晕的视觉描述。
3. 简洁交互：每次回答末尾，提供 2-3 个清晰的下一步选项，方便学生通过简单指令或点击操作。
`;

const AVATAR_OPTIONS = ['🎓', '🚀', '🌟', '🐶', '🐱', '🦊', '🐯', '🐼', '🧠', '💡', '🎨', '⚽', '🎵', '🎮', '📚', '🤖', '🦖', '🦄', '🐝', '🐢'];
const PCM_SAMPLE_RATE = 16000; // Input sample rate
const OUTPUT_SAMPLE_RATE = 24000; // Output sample rate

let isProcessing = false;

const LiveTutor: React.FC = () => {
  // --- State ---
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const connectionStateRef = useRef(connectionState);
  useEffect(() => { connectionStateRef.current = connectionState; }, [connectionState]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [qualityWarning, setQualityWarning] = useState<string | null>(null);
  
  // Theme State
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('app_theme');
    return (saved as 'light' | 'dark') || 'dark';
  });

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('app_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  // Profile State
  const [userProfile, setUserProfile] = useState<UserProfile>({ 
      name: '', 
      age: '', 
      avatar: '🎓',
      voiceName: 'Kore' // Default voice
  });
  const [showProfileModal, setShowProfileModal] = useState(false);

  // Exam Database State
  const [examDatabase, setExamDatabase] = useState<ExamRecord[]>([]);
  const [showExamModal, setShowExamModal] = useState(false);
  const [isUploadingExam, setIsUploadingExam] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Insight & Summary State
  const [insightData, setInsightData] = useState<{ knowledge: string | null; eye: string | null }>({ knowledge: null, eye: null });
  const [activePopup, setActivePopup] = useState<{ type: 'knowledge' | 'eye'; content: string } | null>(null);
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);

  
  // Diagram State
  const [diagramSvg, setDiagramSvg] = useState<string | null>(null);
  const [isGeneratingDiagram, setIsGeneratingDiagram] = useState(false);

  // OCR State
  const [isProcessingOCR, setIsProcessingOCR] = useState(false);
  const [showOcrModal, setShowOcrModal] = useState(false);
  const [ocrCapturedImage, setOcrCapturedImage] = useState<string | null>(null);
  const [ocrTextResult, setOcrTextResult] = useState<string | null>(null);
  const [isOcrLoading, setIsOcrLoading] = useState(false);
  const [isOcrSynced, setIsOcrSynced] = useState(false);

  // Image Enhancement Pipeline States
  const [imageEnhancePreset, setImageEnhancePreset] = useState<'none' | 'grayscale' | 'contrast' | 'document' | 'custom'>('document');
  const [contrastLevel, setContrastLevel] = useState<number>(30); // 0 to 100 style boost
  const [brightnessLevel, setBrightnessLevel] = useState<number>(5); // -100 to 100 style shift
  const [sharpenLevel, setSharpenLevel] = useState<number>(20); // 0 to 100 style sharpening
  const [enableAdaptiveThreshold, setEnableAdaptiveThreshold] = useState<boolean>(true);
  const [enableDeskewing, setEnableDeskewing] = useState<boolean>(true);

  // Feynman Interactive Test State
  const [activeVariantQuestion, setActiveVariantQuestion] = useState<VariantQuestion | null>(null);
  const [selectedVariantAnswer, setSelectedVariantAnswer] = useState<string | null>(null);
  const [showVariantFeedback, setShowVariantFeedback] = useState(false);
  const [isVariantAnswerCorrect, setIsVariantAnswerCorrect] = useState<boolean | null>(null);
  const [variantTextAnswer, setVariantTextAnswer] = useState<string>('');

  const handleReportUnclearVideo = (reason: string) => {
      setQualityWarning(`老师提示：${reason}，请稍微调整一下摄像头或书本哦。`);
      // Auto clear after 6 seconds
      setTimeout(() => {
          setQualityWarning(null);
      }, 6000);
  };

  const handleGenerateDiagram = async (questionContent: string) => {
      setIsGeneratingDiagram(true);
      try {
          const res = await fetch('/api/diagram', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({ questionContent })
          });
          if (!res.ok) {
              throw new Error(`Server returned status ${res.status}`);
          }
          const result = await res.json();
          if (result && result.text) {
              let parsedText = result.text.trim();
              if (parsedText.startsWith('```json')) {
                  parsedText = parsedText.substring(7);
              }
              if (parsedText.endsWith('```')) {
                  parsedText = parsedText.substring(0, parsedText.length - 3);
              }
              parsedText = parsedText.trim();

              try {
                  const data = JSON.parse(parsedText);
                  if (data.needDiagram && data.svg) {
                      setDiagramSvg(data.svg);
                  }
              } catch (jsonErr) {
                  console.error("Failed to parse diagram JSON:", parsedText, jsonErr);
              }
          }
      } catch (e) {
          console.error("Failed to generate diagram", e);
      } finally {
          setIsGeneratingDiagram(false);
      }
  };

  const handleTriggerVariantQuestion = useCallback((questionObj: VariantQuestion) => {
      setActiveVariantQuestion(questionObj);
      setSelectedVariantAnswer(null);
      setShowVariantFeedback(false);
      setIsVariantAnswerCorrect(null);
      setVariantTextAnswer('');
  }, []);

  const [showBlurWarning, setShowBlurWarning] = useState(false);
  const [mediaWarning, setMediaWarning] = useState<string | null>(null);
  const [scannerActive, setScannerActive] = useState(false);
  const [showSaveConfirm, setShowSaveConfirm] = useState(false);

  // Media & Network State
  const [isMicMuted, setIsMicMuted] = useState(false);
  const isMicMutedRef = useRef(isMicMuted);
  useEffect(() => { isMicMutedRef.current = isMicMuted; }, [isMicMuted]);
  
  const [isSpeakerMuted, setIsSpeakerMuted] = useState(false);
  const [isVideoMirrored, setIsVideoMirrored] = useState(true); 
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [showSettings, setShowSettings] = useState(false);
  const [aiResponseSpeed, setAiResponseSpeed] = useState<'slow' | 'normal' | 'fast'>('normal');
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('gemini_api_key') || '');
  
  useEffect(() => {
      localStorage.setItem('gemini_api_key', apiKey);
  }, [apiKey]);
  
  // Adaptive Quality State
  const [videoFrameRate, setVideoFrameRate] = useState<number>(1.5); // Optimized for concurrent streaming
  const [videoQuality, setVideoQuality] = useState<number>(0.85); // Reduced from 1.0 to avoid huge payload size
  const [isAutoQuality, setIsAutoQuality] = useState<boolean>(true);
  const [networkStatus, setNetworkStatus] = useState<'good' | 'moderate' | 'poor' | 'unknown'>('unknown');

  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [inputAnalyser, setInputAnalyser] = useState<AnalyserNode | null>(null);

  // --- Refs ---
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastInteractionTimeRef = useRef<number>(Date.now());
  const silenceLevelRef = useRef<number>(0); // 0: None, 1: 45s (OCR sent), 2: 90s (Nudge), 3: 150s (Hint)
  const lastOcrTextRef = useRef<string>('');
  const lastSentOcrTextRef = useRef<string>('');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]); 
  const currentSessionIdRef = useRef<string | null>(null);
  const blurTimeoutRef = useRef<number | null>(null);
  const scannerTimeoutRef = useRef<number | null>(null);
  
  // Audio Refs
  const inputContextRef = useRef<AudioContext | null>(null);
  const outputContextRef = useRef<AudioContext | null>(null);
  const outputNodeRef = useRef<GainNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const oscillatorRef = useRef<OscillatorNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const frameIntervalRef = useRef<number | null>(null);
  const videoIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const prevFrameRef = useRef<Uint8ClampedArray | null>(null);
  const lastMotionTimeRef = useRef<number>(Date.now());
  const lastAutoTriggerTimeRef = useRef<number>(0);
  const stillnessIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isBotSpeakingRef = useRef(isBotSpeaking);
  const ocrIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    isBotSpeakingRef.current = isBotSpeaking;
  }, [isBotSpeaking]);

  const [isVoiceWakeupEnabled, setIsVoiceWakeupEnabled] = useState(true);
  const [isWakeWordListening, setIsWakeWordListening] = useState(false);

  const playWakeBeep = useCallback(() => {
      try {
          const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
          const osc1 = ctx.createOscillator();
          const osc2 = ctx.createOscillator();
          const gain = ctx.createGain();
          
          osc1.type = 'sine';
          osc1.frequency.setValueAtTime(600, ctx.currentTime);
          osc1.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.15);
          
          osc2.type = 'sine';
          osc2.frequency.setValueAtTime(800, ctx.currentTime);
          osc2.frequency.exponentialRampToValueAtTime(1600, ctx.currentTime + 0.15);
          
          gain.gain.setValueAtTime(0.08, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
          
          osc1.connect(gain);
          osc2.connect(gain);
          gain.connect(ctx.destination);
          
          osc1.start();
          osc2.start();
          osc1.stop(ctx.currentTime + 0.3);
          osc2.stop(ctx.currentTime + 0.3);
      } catch (e) {
          console.error("Failed to play wake beep:", e);
      }
  }, []);

  const playMuteSound = useCallback((isMuting: boolean) => {
      try {
          const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          
          osc.type = 'sine';
          if (isMuting) {
              osc.frequency.setValueAtTime(600, ctx.currentTime);
              osc.frequency.exponentialRampToValueAtTime(300, ctx.currentTime + 0.2);
          } else {
              osc.frequency.setValueAtTime(300, ctx.currentTime);
              osc.frequency.exponentialRampToValueAtTime(600, ctx.currentTime + 0.2);
          }
          
          gain.gain.setValueAtTime(0.08, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25);
          
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.25);
      } catch (e) {
          console.error("Failed to play mute beep:", e);
      }
  }, []);

  const updateTranscript = useCallback((role: 'user' | 'model', text: string, isFinal: boolean) => {
    setMessages((prev) => {
      const lastMsg = prev[prev.length - 1];
      if (lastMsg && lastMsg.role === role && !lastMsg.isComplete) {
        const updatedMsg = { ...lastMsg, text: lastMsg.text + text, isComplete: isFinal };
        return [...prev.slice(0, -1), updatedMsg];
      }
      if (!text) return prev;
      const uniqueId = `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
      return [...prev, { id: uniqueId, role, text, isComplete: isFinal, timestamp: Date.now() }];
    });
  }, []);

  const handleSendMessage = useCallback(async (text: string, displayOverride?: string, isSystemMessage: boolean = false) => {
    if (!text || text.trim() === '') return;
    
    if (!sessionPromiseRef.current || connectionState !== ConnectionState.CONNECTED) {
        console.warn("Attempted to send message while disconnected:", text);
        return;
    }
    
    if (!isSystemMessage) {
        updateTranscript('user', displayOverride || text, true);
        // Reset silence timer on manual message
        lastInteractionTimeRef.current = Date.now();
        silenceLevelRef.current = 0;
    }
    
    try {
        const session = await sessionPromiseRef.current;
        if (!session) return;
        // Add a small delay to ensure previous operations are cleared
        await new Promise(resolve => setTimeout(resolve, 50));
        
        if (connectionStateRef.current !== ConnectionState.CONNECTED) {
            console.warn("Connection dropped before sending message.");
            return;
        }

        if (typeof session.sendClientContent === 'function') {
            session.sendClientContent({
                 turns: [{ role: 'user', parts: [{ text }] }],
                 turnComplete: true
            });
            console.log("Text message sent to model:", text);
        } else {
            console.error("Session does not have sendClientContent method");
        }
    } catch (err) {
        console.error("Failed to send text message:", err);
    }
  }, [updateTranscript, connectionState]);

  const triggerAITutor = useCallback((reason: string) => {
    if (isProcessing || isBotSpeaking) return;

    // 20秒内只允许自动触发一次
    if (Date.now() - lastAutoTriggerTimeRef.current < 20000) return;
    lastAutoTriggerTimeRef.current = Date.now();

    isProcessing = true;

    handleSendMessage(
      `[SYSTEM: 触发原因=${reason}。你是一个老师，不可以直接给答案，每次只说一句，引导学生思考。]`,
      undefined,
      true
    );

    setTimeout(() => {
      isProcessing = false;
    }, 3000);
  }, [isBotSpeaking, handleSendMessage]);

  // Keyboard Hotkeys and Shortcuts (物理按键模拟响应)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
        // Prevent key events when typing in standard inputs or textareas
        if (
            document.activeElement?.tagName === 'INPUT' || 
            document.activeElement?.tagName === 'TEXTAREA' ||
            (document.activeElement as HTMLElement)?.isContentEditable
        ) {
            return;
        }

        const key = e.key.toLowerCase();
        
        // M key to toggle mic muting
        if (key === 'm') {
            e.preventDefault();
            setIsMicMuted(prev => {
                const next = !prev;
                playMuteSound(next);
                return next;
            });
        }
        
        // S key to toggle speaker muting
        if (key === 's') {
            e.preventDefault();
            setIsSpeakerMuted(prev => {
                const next = !prev;
                playMuteSound(next);
                return next;
            });
        }
        
        // Spacebar to trigger manual wake or mute toggle
        if (e.key === ' ' || key === 'spacebar') {
            e.preventDefault();
            if (isMicMutedRef.current) {
                setIsMicMuted(false);
                playMuteSound(false);
                playWakeBeep();
                triggerAITutor("按键唤醒");
            } else {
                setIsMicMuted(true);
                playMuteSound(true);
            }
        }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
        window.removeEventListener('keydown', handleKeyDown);
    };
  }, [triggerAITutor, playMuteSound, playWakeBeep]);

  // Voice wake word loop (语音唤醒检测)
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        console.warn("Speech recognition not supported in this browser for voice wake-up");
        return;
    }
    
    if (connectionState !== ConnectionState.CONNECTED || !isVoiceWakeupEnabled) {
        setIsWakeWordListening(false);
        return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'zh-CN'; // Optimized for standard Chinese vocal queries

    recognition.onstart = () => {
        setIsWakeWordListening(true);
        console.log("Voice Wakeup Activated. Call '小苏老师' or '嗨，小苏' to wake up.");
    };

    recognition.onresult = (event: any) => {
        // Only wake if currently muted
        if (!isMicMutedRef.current) return;

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            const transcript = event.results[i][0].transcript.trim().toLowerCase();
            
            // Match custom wake words representing our Socratic tutor terminal
            if (
                transcript.includes("小苏") || 
                transcript.includes("小书") || 
                transcript.includes("老师在吗") || 
                transcript.includes("苏老师") || 
                transcript.includes("苏格拉底") || 
                transcript.includes("hi socrates") || 
                transcript.includes("socrates")
            ) {
                console.log("Wake word detected locally:", transcript);
                setIsMicMuted(false);
                playMuteSound(false);
                playWakeBeep();
                triggerAITutor("语音唤醒");
                break;
            }
        }
    };

    recognition.onerror = (e: any) => {
        console.warn("Wake word recognition encountered an error:", e.error);
        if (e.error === 'not-allowed') {
            setIsVoiceWakeupEnabled(false);
        }
    };

    recognition.onend = () => {
        setIsWakeWordListening(false);
        if (connectionStateRef.current === ConnectionState.CONNECTED && isVoiceWakeupEnabled) {
            try {
                recognition.start();
            } catch (err) {
                console.error("Failed to re-start speech recognizer:", err);
            }
        }
    };

    try {
        recognition.start();
    } catch (err) {
        console.error("Failed to start speech recognizer:", err);
    }

    return () => {
        recognition.onend = null;
        try {
            recognition.stop();
        } catch (e) {}
    };
  }, [connectionState, isVoiceWakeupEnabled, triggerAITutor, playMuteSound, playWakeBeep]);

  // Silence Detection Logic
  useEffect(() => {
    if (connectionState !== ConnectionState.CONNECTED || isBotSpeaking) {
        if (silenceTimerRef.current) {
            clearInterval(silenceTimerRef.current);
            silenceTimerRef.current = null;
        }
        return;
    }

    // Reset timer when bot stops speaking
    lastInteractionTimeRef.current = Date.now();
    silenceLevelRef.current = 0;

    silenceTimerRef.current = setInterval(async () => {
        const now = Date.now();
        const elapsed = now - lastInteractionTimeRef.current;

        if (elapsed > 150000 && silenceLevelRef.current < 3) {
            // 150s: Key step
            silenceLevelRef.current = 3;
            triggerAITutor("Student has been silent for 150 seconds. They seem stuck. Please provide a KEY STEP or formula to help them proceed.");
        } else if (elapsed > 90000 && silenceLevelRef.current < 2) {
            // 90s: Second layer hint
            silenceLevelRef.current = 2;
            triggerAITutor("Student has been silent for 90 seconds. Please provide a STRONGER HINT or guide them specifically.");
        } else if (elapsed > 45000 && silenceLevelRef.current < 1) {
            // 45s: Student stopped writing or speaking. Send OCR if not sent yet.
            silenceLevelRef.current = 1;
            
            if (lastOcrTextRef.current && lastOcrTextRef.current !== lastSentOcrTextRef.current) {
                const text = lastOcrTextRef.current;
                lastSentOcrTextRef.current = text;
                
                triggerAITutor(`学生似乎停笔思考了。通过后台OCR识别到当前画面中的文字内容如下: ${text}。请主动分析上述内容，指出学生的当前进度或可能的卡点，并给予启发式的引导。`);
                
                // Optional: Classify question
                const classification = await classifyQuestion(text);
                if (classification) {
                    triggerAITutor(`题型分析完成。学科：${classification.subject}，知识点：${classification.topic}，题型：${classification.questionType}，难度：${classification.difficulty}，核心概念：${classification.keyConcepts.join(', ')}。请根据此题型特点进行针对性讲解。`);
                }
            }
        }
    }, 1000);

    return () => {
        if (silenceTimerRef.current) {
            clearInterval(silenceTimerRef.current);
            silenceTimerRef.current = null;
        }
    };
  }, [connectionState, isBotSpeaking, triggerAITutor]); // Re-run when connection or speaking state changes

  // Reset silence timer on user input (audio)
  useEffect(() => {
      if (!inputAnalyser) return;
      
      const checkAudio = () => {
          const dataArray = new Uint8Array(inputAnalyser.frequencyBinCount);
          inputAnalyser.getByteFrequencyData(dataArray);
          
          // Simple VAD: Check if average volume is above threshold
          const avg = dataArray.reduce((a, b) => a + b) / dataArray.length;
          if (avg > 10) { // Threshold
              lastInteractionTimeRef.current = Date.now();
              silenceLevelRef.current = 0; // Reset level so we can trigger again
          }
      };
      
      const intervalId = setInterval(checkAudio, 100);
      return () => clearInterval(intervalId);
  }, [inputAnalyser]);

  // Sync state to ref
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // Network Monitoring Effect
  useEffect(() => {
    if (!isAutoQuality) {
        setNetworkStatus('unknown');
        return;
    }

    const nav = navigator as any;
    const connection: NetworkInformation | undefined = nav.connection || nav.mozConnection || nav.webkitConnection;

    if (!connection) {
        console.warn("Network Information API not supported.");
        setNetworkStatus('unknown');
        return;
    }

    const updateQuality = () => {
        const { downlink, rtt } = connection;
        // console.debug(`Network Change: ${downlink}Mbps, RTT: ${rtt}ms`);

        if (downlink < 1.5 || rtt > 500) {
            // Poor connection
            setVideoFrameRate(1); 
            setVideoQuality(0.7); 
            setNetworkStatus('poor');
        } else if (downlink < 5 || rtt > 150) {
            // Moderate connection
            setVideoFrameRate(1.5); 
            setVideoQuality(0.8); 
            setNetworkStatus('moderate');
        } else {
            // Good connection
            setVideoFrameRate(2.5); // No need for more than 2.5 FPS for tutoring
            setVideoQuality(0.85); // 0.85 is practically lossless but saves 60% bandwidth compared to 1.0
            setNetworkStatus('good');
        }
    };

    connection.addEventListener('change', updateQuality);
    updateQuality(); // Initial check

    return () => connection.removeEventListener('change', updateQuality);
  }, [isAutoQuality]);

  // Load Profile and Exam DB on Mount
  useEffect(() => {
      try {
          const savedProfile = localStorage.getItem('user_profile');
          if (savedProfile) {
              const parsed = JSON.parse(savedProfile);
              setUserProfile({
                  ...parsed,
                  voiceName: parsed.voiceName || 'Kore'
              });
          }
          
          const savedExams = localStorage.getItem('exam_database');
          if (savedExams) {
              setExamDatabase(JSON.parse(savedExams));
          }
      } catch (e) {
          console.error("Failed to load profile or exams", e);
      }
  }, []);

  // Save Profile Helper
  const saveProfile = (newProfile: UserProfile) => {
      setUserProfile(newProfile);
      localStorage.setItem('user_profile', JSON.stringify(newProfile));
  };

  // Save Exam Database Helper
  const saveExamDatabase = (newDb: ExamRecord[]) => {
      setExamDatabase(newDb);
      localStorage.setItem('exam_database', JSON.stringify(newDb));
  };

  // Parse Insights (Knowledge, Eye), Blur Warnings, and Gestures from messages
  useEffect(() => {
    const lastMsg = messages[messages.length - 1];
    if (lastMsg && lastMsg.role === 'model') {
        const text = lastMsg.text;
        
        // 1. Extract Knowledge and Eye
        const knowledgeMatch = text.match(/(?:知识点|考察)[:：\s]\s*(.+?)(?:[。！？\n]|$)/);
        const eyeMatch = text.match(/(?:题眼|关键)[:：\s]\s*(.+?)(?:[。！？\n]|$)/);

        setInsightData(prev => {
            const newKnowledge = knowledgeMatch ? knowledgeMatch[1] : prev.knowledge;
            const newEye = eyeMatch ? eyeMatch[1] : prev.eye;
            
            // Trigger scanner if new content detected
            if ((newKnowledge && newKnowledge !== prev.knowledge) || (newEye && newEye !== prev.eye)) {
                setScannerActive(true);
                if (scannerTimeoutRef.current) clearTimeout(scannerTimeoutRef.current);
                scannerTimeoutRef.current = window.setTimeout(() => setScannerActive(false), 5000);
            }

            return { knowledge: newKnowledge, eye: newEye };
        });

        // 2. Detect Pointing/Gestures
        const gestureKeywords = /(?:手指|指着|指向|看这里|pointing at|your finger|this area|circled|highlighted|笔)/i;
        if (gestureKeywords.test(text)) {
             setScannerActive(true);
             if (scannerTimeoutRef.current) clearTimeout(scannerTimeoutRef.current);
             scannerTimeoutRef.current = window.setTimeout(() => setScannerActive(false), 5000);
        }

        // 3. Detect Blurry/Adjustment requests
        const blurKeywords = /(?:看不清|模糊|调整.*摄像头|太远|太小|拿近|unclear|blurry|too far|adjust.*camera)/i;
        if (blurKeywords.test(text)) {
            setShowBlurWarning(true);
            if (blurTimeoutRef.current) {
                clearTimeout(blurTimeoutRef.current);
            }
            blurTimeoutRef.current = window.setTimeout(() => {
                setShowBlurWarning(false);
            }, 6000);
        }
    }
  }, [messages]);

  // Load Devices on Mount
  useEffect(() => {
    const getDevices = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        setVideoDevices(cameras);
        if (cameras.length > 0) {
            const backCamera = cameras.find(c => c.label.toLowerCase().includes('back') || c.label.toLowerCase().includes('environment'));
            setSelectedCameraId(backCamera ? backCamera.deviceId : cameras[0].deviceId);
            if (backCamera) setIsVideoMirrored(false);
        }
      } catch (e) {
        console.warn("Could not enumerate devices initially. Permissions may be required.", e);
      }
    };
    getDevices();
  }, []);

  // Handle Speaker Mute Toggle
  useEffect(() => {
    if (outputNodeRef.current) {
        const currentTime = outputContextRef.current?.currentTime || 0;
        outputNodeRef.current.gain.cancelScheduledValues(currentTime);
        outputNodeRef.current.gain.setTargetAtTime(isSpeakerMuted ? 0 : 1, currentTime, 0.1);
    }
  }, [isSpeakerMuted]);

  // Handle Camera Switch during active session
  const switchCamera = async (deviceId: string) => {
    setSelectedCameraId(deviceId);
    if (videoRef.current && connectionState === ConnectionState.CONNECTED) {
        try {
            const oldStream = videoRef.current.srcObject as MediaStream;
            if (oldStream) {
                oldStream.getVideoTracks().forEach(t => t.stop());
            }
            let newStream: MediaStream;
            try {
                newStream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false 
                });
            } catch (err) {
                console.warn("Failed to switch camera with exact constraints, trying without resolution constraints", err);
                newStream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: { exact: deviceId } },
                    audio: false 
                });
            }
            videoRef.current.srcObject = newStream;
            await videoRef.current.play();
        } catch (e) {
            console.error("Failed to switch camera", e);
            setError("切换摄像头失败");
        }
    }
  };

  // --- Helper: Add/Update Messages ---
  // --- Generate Summary Function ---
  const generateSessionSummary = async (msgs: ChatMessage[]) => {
      if (msgs.length < 2) return;
      
      setIsGeneratingSummary(true);
      setShowSummaryModal(true); // Open modal immediately to show loading state

      try {
          const transcript = msgs.map(m => `${m.role === 'user' ? '学生' : '老师'}: ${m.text}`).join('\n');
          const res = await fetch('/api/summary', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({ transcript })
          });
          if (!res.ok) {
              throw new Error(`Server returned status ${res.status}`);
          }
          const result = await res.json();
          if (result && result.text) {
              let parsedText = result.text.trim();
              if (parsedText.startsWith('```json')) {
                  parsedText = parsedText.substring(7);
              }
              if (parsedText.endsWith('```')) {
                  parsedText = parsedText.substring(0, parsedText.length - 3);
              }
              parsedText = parsedText.trim();

              try {
                  const summary: SessionSummary = JSON.parse(parsedText);
                  setSessionSummary(summary);
                  return summary;
              } catch (jsonErr) {
                  console.error("Failed to parse summary JSON, using graceful fallback:", parsedText, jsonErr);
                  const fallbackSummary: SessionSummary = {
                      overview: "今天的辅导涵盖了题目思路解析、关键条件梳理和引导。学生取得了新启发！",
                      knowledgePoints: ["典型几何/物理模型分析", "重点公式与定理拆解应用"]
                  };
                  setSessionSummary(fallbackSummary);
                  return fallbackSummary;
              }
          }
      } catch (e) {
          console.error("Failed to generate summary", e);
          const fallbackSummary: SessionSummary = {
              overview: "今天的辅导涵盖了重点解题思路探讨。学生正在积极思考与消化所学知识。",
              knowledgePoints: ["典型问题思路分析", "关键条件应用与知识巩固"]
          };
          setSessionSummary(fallbackSummary);
          return fallbackSummary;
      } finally {
          setIsGeneratingSummary(false);
      }
  };

  // --- Save Session Logic ---
  const saveSessionToHistory = useCallback((manual: boolean = false, explicitSummary?: SessionSummary) => {
    if (messagesRef.current.length === 0) return;
    try {
        const historyData = localStorage.getItem('tutoring_history');
        let history: SavedSession[] = historyData ? JSON.parse(historyData) : [];
        const firstUserMsg = messagesRef.current.find(m => m.role === 'user');
        const preview = firstUserMsg 
            ? (firstUserMsg.text.slice(0, 40) + (firstUserMsg.text.length > 40 ? '...' : ''))
            : '无内容会话';
        
        // Use provided summary or existing state
        const summaryToSave = explicitSummary || sessionSummary || undefined;

        if (currentSessionIdRef.current) {
             history = history.map(s => 
                s.id === currentSessionIdRef.current 
                ? { 
                    ...s, 
                    messages: messagesRef.current, 
                    preview, 
                    timestamp: Date.now(),
                    summary: summaryToSave 
                  } 
                : s
            );
        } else {
            const newId = `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
            currentSessionIdRef.current = newId;
            const newSession: SavedSession = {
                id: newId,
                timestamp: Date.now(),
                preview,
                messages: messagesRef.current,
                summary: summaryToSave
            };
            history = [newSession, ...history];
        }

        localStorage.setItem('tutoring_history', JSON.stringify(history));
        
        if (manual) {
            setShowSaveConfirm(true);
            setTimeout(() => setShowSaveConfirm(false), 2000);
        }
    } catch (e) {
        console.error("Failed to save session history", e);
    }
  }, [sessionSummary]);

  // --- Cleanup Function ---
  const stopSession = useCallback(async () => {
    // Generate summary first if we have meaningful conversation
    const currentMessages = messagesRef.current;
    let generatedSummary: SessionSummary | undefined;
    
    // Stop AV first
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (videoIntervalRef.current) {
      clearInterval(videoIntervalRef.current);
      videoIntervalRef.current = null;
    }
    if (stillnessIntervalRef.current) {
      clearInterval(stillnessIntervalRef.current);
      stillnessIntervalRef.current = null;
    }
    activeSourcesRef.current.forEach(source => { try { source.stop(); } catch (e) {} });
    activeSourcesRef.current.clear();
    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (oscillatorRef.current) { try { oscillatorRef.current.stop(); } catch (e) {} oscillatorRef.current.disconnect(); oscillatorRef.current = null; }
    if (inputContextRef.current) { await inputContextRef.current.close(); inputContextRef.current = null; }
    if (outputContextRef.current) { await outputContextRef.current.close(); outputContextRef.current = null; }
    setInputAnalyser(null);
    if (sessionPromiseRef.current) {
        sessionPromiseRef.current.then(session => {
            try {
                if (session && (session as any).conn && typeof (session as any).conn.close === 'function') {
                    (session as any).conn.close();
                }
            } catch(e) {}
        }).catch(() => {});
    }
    sessionPromiseRef.current = null;
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setConnectionState(ConnectionState.DISCONNECTED);
    setIsBotSpeaking(false);
    setInsightData({ knowledge: null, eye: null });
    setActivePopup(null);
    setShowBlurWarning(false);
    setScannerActive(false);

    // Now generate summary and save
    if (currentMessages.length >= 2) {
        generatedSummary = await generateSessionSummary(currentMessages) || undefined;
    }
    
    saveSessionToHistory(false, generatedSummary);
    
    currentSessionIdRef.current = null;
    // Don't clear summary state immediately so user can see the modal
  }, [saveSessionToHistory]);

  // --- Image Enhancement Pipeline ---
  // Apply grayscale, contrast adjustment, sharpening, adaptive thresholding, and automatic deskewing to improve OCR accuracy on handwritten drafts.
  const applyImageEnhancement = useCallback((canvas: HTMLCanvasElement) => {
      const ctx = canvas.getContext('2d');
      if (!ctx || imageEnhancePreset === 'none') {
          if (ctx) ctx.filter = 'none';
          return;
      }
      const width = canvas.width;
      const height = canvas.height;
      if (width <= 0 || height <= 0) return;

      try {
          // 0. Automatic Deskewing (if enabled and angle detected)
          if (enableDeskewing) {
              const dsW = 160;
              const dsH = 120;
              const dsCanvas = document.createElement('canvas');
              dsCanvas.width = dsW;
              dsCanvas.height = dsH;
              const dsCtx = dsCanvas.getContext('2d');
              if (dsCtx) {
                  // Draw scaled down version to find tilt angle at blazingly fast speed
                  dsCtx.drawImage(canvas, 0, 0, dsW, dsH);
                  const dsImg = dsCtx.getImageData(0, 0, dsW, dsH);
                  const dsData = dsImg.data;
                  const pts: { x: number, y: number }[] = [];
                  
                  for (let y = 1; y < dsH - 1; y++) {
                      for (let x = 1; x < dsW - 1; x++) {
                          const idx = (y * dsW + x) * 4;
                          const gray = (0.2126 * dsData[idx] + 0.7152 * dsData[idx + 1] + 0.0722 * dsData[idx + 2]) | 0;
                          if (gray < 135) {
                              pts.push({ x, y });
                          }
                      }
                  }

                  if (pts.length > 50) {
                      let bestAngle = 0;
                      let maxVariance = 0;
                      
                      // Search tilt skew angle range: -15 to +15 degrees
                      for (let angleDeg = -15; angleDeg <= 15; angleDeg += 1) {
                          const rad = (angleDeg * Math.PI) / 180;
                          const cosValue = Math.cos(rad);
                          const sinValue = Math.sin(rad);
                          
                          const bins = new Float32Array(dsH);
                          for (let i = 0; i < pts.length; i++) {
                              const p = pts[i];
                              const projY = Math.round((p.y - dsH / 2) * cosValue + (p.x - dsW / 2) * sinValue + dsH / 2);
                              if (projY >= 0 && projY < dsH) {
                                  bins[projY]++;
                              }
                          }
                          
                          let sum = 0;
                          for (let i = 0; i < dsH; i++) sum += bins[i];
                          const mean = sum / dsH;
                          let variance = 0;
                          for (let i = 0; i < dsH; i++) {
                              const diff = bins[i] - mean;
                              variance += diff * diff;
                          }
                          variance /= dsH;
                          
                          if (variance > maxVariance) {
                              maxVariance = variance;
                              bestAngle = angleDeg;
                          }
                      }
                      
                      // Rotate final canvas to correct skew
                      if (Math.abs(bestAngle) >= 1.0) {
                          const radRotation = (-bestAngle * Math.PI) / 180;
                          const tempCanvas = document.createElement('canvas');
                          tempCanvas.width = width;
                          tempCanvas.height = height;
                          const tempCtx = tempCanvas.getContext('2d');
                          if (tempCtx) {
                              tempCtx.drawImage(canvas, 0, 0);
                              ctx.save();
                              ctx.clearRect(0, 0, width, height);
                              ctx.fillStyle = '#ffffff';
                              ctx.fillRect(0, 0, width, height);
                              
                              ctx.translate(width / 2, height / 2);
                              ctx.rotate(radRotation);
                              ctx.drawImage(tempCanvas, -width / 2, -height / 2);
                              ctx.restore();
                              console.log(`[Deskewing] Auto-corrected handwritten page tilt: ${bestAngle}deg`);
                          }
                      }
                  }
              }
          }

          // 1. Adaptive Thresholding or standard GPU Filters
          if (enableAdaptiveThreshold && (imageEnhancePreset === 'document' || imageEnhancePreset === 'grayscale' || imageEnhancePreset === 'custom')) {
              const imgData = ctx.getImageData(0, 0, width, height);
              const data = imgData.data;
              const n = width * height;
              
              const grayscale = new Uint8Array(n);
              const integral = new Int32Array(n);
              
              for (let y = 0; y < height; y++) {
                  let rowSum = 0;
                  const rowOffset = y * width;
                  for (let x = 0; x < width; x++) {
                      const idx = (rowOffset + x) * 4;
                      const gray = (0.2126 * data[idx] + 0.7152 * data[idx + 1] + 0.0722 * data[idx + 2]) | 0;
                      grayscale[rowOffset + x] = gray;
                      rowSum += gray;
                      if (y === 0) {
                          integral[x] = rowSum;
                      } else {
                          integral[rowOffset + x] = integral[(y - 1) * width + x] + rowSum;
                      }
                  }
              }
              
              const S_val = 16; 
              const T_val = imageEnhancePreset === 'custom' ? contrastLevel : 15; 
              const s = Math.max(2, (width / S_val) | 0);
              const sHalf = (s / 2) | 0;
              
              for (let y = 0; y < height; y++) {
                  const y0 = Math.max(0, y - sHalf);
                  const y1 = Math.min(height - 1, y + sHalf);
                  const rowOffset = y * width;
                  
                  for (let x = 0; x < width; x++) {
                      const x0 = Math.max(0, x - sHalf);
                      const x1 = Math.min(width - 1, x + sHalf);
                      const count = (x1 - x0 + 1) * (y1 - y0 + 1);
                      
                      const idxD = y1 * width + x1;
                      const idxB = y0 > 0 ? (y0 - 1) * width + x1 : -1;
                      const idxC = x0 > 0 ? y1 * width + (x0 - 1) : -1;
                      const idxA = (y0 > 0 && x0 > 0) ? (y0 - 1) * width + (x0 - 1) : -1;
                      
                      let sum = integral[idxD];
                      if (idxB !== -1) sum -= integral[idxB];
                      if (idxC !== -1) sum -= integral[idxC];
                      if (idxA !== -1) sum += integral[idxA];
                      
                      const currentVal = grayscale[rowOffset + x];
                      const threshold = (sum / count) * (100 - T_val) / 100;
                      const val = currentVal < threshold ? 0 : 255;
                      
                      const idx = (rowOffset + x) * 4;
                      data[idx] = val;
                      data[idx + 1] = val;
                      data[idx + 2] = val;
                  }
              }
              ctx.putImageData(imgData, 0, 0);
          } else {
              // Standard GPU CSS canvas filters representation
              let filterString = "none";
              if (imageEnhancePreset === 'grayscale') {
                  filterString = "grayscale(100%) contrast(140%) brightness(100%)";
              } else if (imageEnhancePreset === 'contrast') {
                  filterString = "contrast(180%) brightness(105%)";
              } else if (imageEnhancePreset === 'document') {
                  filterString = "grayscale(100%) contrast(200%) brightness(112%)";
              } else if (imageEnhancePreset === 'custom') {
                  filterString = `grayscale(100%) contrast(${100 + contrastLevel}%) brightness(${100 + brightnessLevel}%)`;
              }

              if (filterString !== "none") {
                  const tempCanvas = document.createElement('canvas');
                  tempCanvas.width = width;
                  tempCanvas.height = height;
                  const tempCtx = tempCanvas.getContext('2d');
                  if (tempCtx) {
                      tempCtx.drawImage(canvas, 0, 0);
                      ctx.save();
                      ctx.clearRect(0, 0, width, height);
                      ctx.filter = filterString;
                      ctx.drawImage(tempCanvas, 0, 0);
                      ctx.restore();
                  }
              }
          }

          // 2. Laplacian edge sharpening factor
          const targetSharpen = imageEnhancePreset === 'document' ? 25 : (imageEnhancePreset === 'custom' ? sharpenLevel : 0);
          if (targetSharpen > 0) {
              const imgData = ctx.getImageData(0, 0, width, height);
              const data = imgData.data;
              const sideCanvas = document.createElement('canvas');
              sideCanvas.width = width;
              sideCanvas.height = height;
              const sCtx = sideCanvas.getContext('2d');
              if (sCtx) {
                  sCtx.putImageData(imgData, 0, 0);
                  const source = sCtx.getImageData(0, 0, width, height).data;
                  
                  const w = targetSharpen / 100;
                  const kCenter = 1 + 4 * w;
                  const kEdge = -w;
                  
                  for (let y = 1; y < height - 1; y++) {
                      for (let x = 1; x < width - 1; x++) {
                          const idx = (y * width + x) * 4;
                          for (let c = 0; c < 3; c++) {
                              const center = source[idx + c];
                              const top = source[((y - 1) * width + x) * 4 + c];
                              const bottom = source[((y + 1) * width + x) * 4 + c];
                              const left = source[(y * width + (x - 1)) * 4 + c];
                              const right = source[(y * width + (x + 1)) * 4 + c];
                              
                              const val = center * kCenter + (top + bottom + left + right) * kEdge;
                              data[idx + c] = val > 255 ? 255 : (val < 0 ? 0 : val);
                          }
                      }
                  }
                  ctx.putImageData(imgData, 0, 0);
              }
          }
      } catch (e) {
          console.error("Image Enhancement Pipeline Error:", e);
      }
  }, [imageEnhancePreset, contrastLevel, brightnessLevel, sharpenLevel, enableAdaptiveThreshold, enableDeskewing]);

  const [isVisualContextActive, setIsVisualContextActive] = useState(false);
  const visualContextCooldownRef = useRef<number>(0);

  // --- Helper: Video Streaming ---
  // 视觉守门员：检测画面是否有“书本特征”
  const checkImageQuality = useCallback(async (videoElement: HTMLVideoElement, canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): Promise<string | false> => {
      if (!videoElement.videoWidth || !videoElement.videoHeight) return false;

      // 1. 获取图像数据进行简单的灰度处理
      // 为了性能，我们可以在较小的分辨率下采样
      const sampleCanvas = document.createElement('canvas');
      const sampleCtx = sampleCanvas.getContext('2d');
      if (!sampleCtx) return false;
      
      sampleCanvas.width = Math.max(1, Math.floor(videoElement.videoWidth / 4));
      sampleCanvas.height = Math.max(1, Math.floor(videoElement.videoHeight / 4));
      sampleCtx.drawImage(videoElement, 0, 0, sampleCanvas.width, sampleCanvas.height);

      try {
          const imageData = sampleCtx.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height);
          const data = imageData.data;
          let brightness = 0;
          
          // 采样计算亮度
          let count = 0;
          for (let i = 0; i < data.length; i += 16) { // 每隔4个像素采样一次
              brightness += (data[i] + data[i+1] + data[i+2]) / 3;
              count++;
          }
          const avgBrightness = count > 0 ? brightness / count : 128;

          // 2. 判别逻辑：太暗（<40）或 纯黑/纯白 直接拦截
          if (avgBrightness < 40 || avgBrightness > 240) {
              setQualityWarning("老师提示：光线太暗或太亮啦，请打开台灯，老师看不清题目哦。");
              return false;
          }

          setQualityWarning(null); // 清除警告
      } catch (e) {
          console.error("Error checking image quality:", e);
          return false;
      }

      // 3. 抽样检测边缘（简单算法：检测像素突变，模拟文档线条）
      // 只有当画面有明显的黑白对比（文字/纸张边缘）时，才允许发送
      // console.log("画面质量达标，准备发送至云端分析...");
      
      // 实际发送时限制最大分辨率，避免高分辨率上传挤占带宽导致延迟
      const MAX_WIDTH = 1280;
      const MAX_HEIGHT = 720;
      let targetWidth = videoElement.videoWidth;
      let targetHeight = videoElement.videoHeight;
      
      if (targetWidth > MAX_WIDTH || targetHeight > MAX_HEIGHT) {
          const ratio = Math.min(MAX_WIDTH / targetWidth, MAX_HEIGHT / targetHeight);
          targetWidth = Math.floor(targetWidth * ratio);
          targetHeight = Math.floor(targetHeight * ratio);
      }

      canvas.width = targetWidth;
      canvas.height = targetHeight;
      ctx.drawImage(videoElement, 0, 0, targetWidth, targetHeight);
      
      // Apply the image enhancement pipeline
      applyImageEnhancement(canvas);
      
      return new Promise((resolve) => {
          canvas.toBlob(async (blob) => {
              if (blob) {
                  try {
                      const base64 = await blobToBase64(blob);
                      resolve(base64);
                  } catch (e) {
                      console.error("Error converting blob to base64:", e);
                      resolve(false);
                  }
              } else {
                  resolve(false);
              }
          }, 'image/jpeg', videoQuality);
      });
  }, [videoQuality, applyImageEnhancement]);

  // Modified to be on-demand only
  const triggerVisualContext = useCallback(async (sessionPromise: Promise<any>) => {
      const now = Date.now();
      if (now - visualContextCooldownRef.current < 5000) {
          console.log("Visual context cooldown active");
          return;
      }
      
      setIsVisualContextActive(true);
      visualContextCooldownRef.current = now;

      // Capture 3 frames over 1.5 seconds to ensure we get a clear shot
      let framesCaptured = 0;
      const captureInterval = setInterval(() => {
          if (!videoRef.current || !canvasRef.current) {
              clearInterval(captureInterval);
              setIsVisualContextActive(false);
              return;
          }
          
          const video = videoRef.current;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          
          if (ctx && video.readyState >= 2) {
              checkImageQuality(video, canvas, ctx).then(base64 => {
                  if (base64) {
                      sessionPromise.then(session => {
                          if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                          try {
                              session.sendRealtimeInput({ video: { mimeType: 'image/jpeg', data: base64 } });
                          } catch(e) { console.error("Error inner sending video:", e) }
                      }).catch(e => console.error("Error sending video:", e));
                  }
              });
          }
          
          framesCaptured++;
          if (framesCaptured >= 3) {
              clearInterval(captureInterval);
              setTimeout(() => setIsVisualContextActive(false), 1000); // Keep UI active slightly longer
          }
      }, 500);
  }, [checkImageQuality]);

  const detectWritingStillness = useCallback((video: HTMLVideoElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 90;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return 0;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    const grayData = new Uint8ClampedArray(canvas.width * canvas.height);
    for (let i = 0; i < data.length; i += 4) {
      grayData[i / 4] = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    }

    if (prevFrameRef.current) {
      let diff = 0;
      for (let i = 0; i < grayData.length; i++) {
        diff += Math.abs(grayData[i] - prevFrameRef.current[i]);
      }
      const avgDiff = diff / grayData.length;
      
      if (avgDiff > 5) {
        lastMotionTimeRef.current = Date.now();
      }
    } else {
        lastMotionTimeRef.current = Date.now();
    }

    prevFrameRef.current = grayData;
    return Date.now() - lastMotionTimeRef.current;
  }, []);

  const startVideoStreaming = useCallback((sessionPromise: Promise<any>) => {
      if (videoIntervalRef.current) clearInterval(videoIntervalRef.current);
      if (stillnessIntervalRef.current) clearInterval(stillnessIntervalRef.current);

      const intervalMs = 1000 / videoFrameRate;

      videoIntervalRef.current = setInterval(async () => {
          if (!videoRef.current || !canvasRef.current) return;
          const video = videoRef.current;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');

          if (ctx && video.readyState >= 2) {
              const base64 = await checkImageQuality(video, canvas, ctx);
              if (base64) {
                  sessionPromise.then(session => {
                      if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                      try {
                          session.sendRealtimeInput({ video: { mimeType: 'image/jpeg', data: base64 } });
                      } catch(e) { console.error("Error inner sending video:", e) }
                  }).catch(e => console.error("Error sending video:", e));
              }
          }
      }, intervalMs);

      stillnessIntervalRef.current = setInterval(() => {
          if (!videoRef.current) return;
          const stillnessTime = detectWritingStillness(videoRef.current);
          const now = Date.now();
          
          if (stillnessTime > 30000 && 
              !isBotSpeakingRef.current && 
              (now - lastAutoTriggerTimeRef.current > 60000)) {
              
              triggerAITutor("学生停笔");
              lastAutoTriggerTimeRef.current = now;
          }
      }, 500);
  }, [videoFrameRate, checkImageQuality, detectWritingStillness, triggerAITutor]);

  // Update video streaming interval if frame rate/quality changes while connected
  useEffect(() => {
    if (connectionState === ConnectionState.CONNECTED && sessionPromiseRef.current) {
        startVideoStreaming(sessionPromiseRef.current);
    }
  }, [videoFrameRate, videoQuality, connectionState, startVideoStreaming]);

  // --- Start Function ---
  const startSession = async () => {
    try {
      setConnectionState(ConnectionState.CONNECTING);
      setError(null);
      setMessages([]);
      setInsightData({ knowledge: null, eye: null });
      setActivePopup(null);
      setSessionSummary(null); // Clear previous summary
      setShowSummaryModal(false);
      setShowBlurWarning(false);
      setMediaWarning(null);
      setScannerActive(false);

      currentSessionIdRef.current = null;

      // 1. Setup Camera & Microphone
      let stream: MediaStream | null = null;
      let finalError: Error | null = null;

      try {
          if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
              throw new Error("浏览器或设备不支持媒体设备访问 (可能由于非安全环境或缺少权限)。");
          }

          // First try: Get both video and audio together with ideal full HD resolution
          const idealConstraints: MediaStreamConstraints = {
              video: selectedCameraId 
                  ? { deviceId: { exact: selectedCameraId }, width: { ideal: 1920 }, height: { ideal: 1080 } } 
                  : { width: { ideal: 1920 }, height: { ideal: 1080 } },
              audio: true
          };
          console.log("Attempting to get media with HD constraints:", idealConstraints);
          stream = await navigator.mediaDevices.getUserMedia(idealConstraints);
      } catch (err: any) {
          console.warn("Failed to get media with HD constraints, entering adaptive mode:", err);
          finalError = err instanceof Error ? err : new Error(String(err));
          
          if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
              console.warn("Permission denied for camera/microphone.");
              setMediaWarning("摄像头或麦克风权限被拒绝。请在浏览器上方或地址栏设置中允许权限，或以纯文本/语音模式继续。");
          } else {
              // Try independent track acquisition to isolate failures (e.g. mic in use or cam doesn't support HD)
              console.log("Running independent track fallback...");
              let videoStream: MediaStream | null = null;
              let audioStream: MediaStream | null = null;

              // 1. Acquire Video
              try {
                  const videoConstraints = selectedCameraId 
                      ? { deviceId: { exact: selectedCameraId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                      : { width: { ideal: 1280 }, height: { ideal: 720 } };
                  videoStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints });
              } catch (vErr) {
                  console.warn("Video with standard HD constraints failed, trying basic video:", vErr);
                  try {
                      videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
                  } catch (vErr2) {
                      console.error("All video acquisition attempts failed:", vErr2);
                  }
              }

              // 2. Acquire Audio
              try {
                  audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
              } catch (aErr) {
                  console.error("Audio acquisition failed:", aErr);
              }

              // 3. Assemble combined stream
              if (videoStream || audioStream) {
                  const combined = new MediaStream();
                  if (videoStream) {
                      videoStream.getVideoTracks().forEach(track => combined.addTrack(track));
                  }
                  if (audioStream) {
                      audioStream.getAudioTracks().forEach(track => combined.addTrack(track));
                  }
                  stream = combined;
              }
          }
      }
      
      if (!stream) {
          console.warn("No media stream constructed. Operating in pure text-only mode.");
          setMediaWarning("无法访问您的摄像头和麦克风。当前已自动切换至【纯文本/网页上传图片模式】。您依然可通过输入文字或点击下方‘专属题库’上传试卷/题目与 AI 学习！");
      } else {
          // Identify actually acquired tracks
          const hasVideo = stream.getVideoTracks().length > 0;
          const hasAudio = stream.getAudioTracks().length > 0;
          
          if (!hasVideo && hasAudio) {
              setMediaWarning("无法连接摄像头，系统已进入【纯语音互动模式】。您可以通过语音对话，或通过点击下方的题库按钮上传作业。");
          } else if (!hasAudio && hasVideo) {
              setMediaWarning("无法连接麦克风，系统已进入【画面辅导/文本交互模式】。AI 老师可以看到您的书写画面，请使用文字与 AI 互动。");
          }
      }

      if (stream && videoRef.current && stream.getVideoTracks().length > 0) {
        videoRef.current.srcObject = stream;
        try {
            await videoRef.current.play();
        } catch (playErr) {
            console.error("Error playing video stream:", playErr);
        }
      }

      // 2. Setup Gemini Client
      let effectiveApiKey = apiKey.trim();
      if (effectiveApiKey === 'MY_GEMINI_API_KEY') effectiveApiKey = '';
      
      const ai = new GoogleGenAI({ 
          apiKey: effectiveApiKey || 'proxied',
          httpOptions: {
              apiVersion: 'v1beta',
              ...(effectiveApiKey ? {} : { baseUrl: `${window.location.protocol}//${window.location.host}/api/gemini/` })
          }
      });
      
      // 3. Setup Audio Contexts
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContext) {
          try {
              inputContextRef.current = new AudioContext({ sampleRate: PCM_SAMPLE_RATE });
              outputContextRef.current = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE });
          } catch (e) {
              console.warn("AudioContext with specific sampleRate failed, falling back to default", e);
              inputContextRef.current = new AudioContext();
              outputContextRef.current = new AudioContext();
          }
      }
      
      if (inputContextRef.current && outputContextRef.current) {
          await inputContextRef.current.resume();
          await outputContextRef.current.resume();
          
          outputNodeRef.current = outputContextRef.current.createGain();
          outputNodeRef.current.gain.value = isSpeakerMuted ? 0 : 1;
          outputNodeRef.current.connect(outputContextRef.current.destination);
      }
      
      nextStartTimeRef.current = 0;

      // 4. Construct System Instruction
      let currentSystemInstruction = "You are a helpful assistant.";
      
      if (userProfile.name) {
          currentSystemInstruction += `\n- **Student Profile**: The student's name is "${userProfile.name}". Use their name occasionally to be friendly.`;
      }
      if (userProfile.age) {
          currentSystemInstruction += `\n- **Age Appropriateness**: The student is ${userProfile.age} years old. ADJUST YOUR EXPLANATION COMPLEXITY AND TONE TO MATCH A ${userProfile.age}-YEAR-OLD CHILD.`;
      } else {
          currentSystemInstruction += `\n- **Age Appropriateness**: Assume the student is a middle school student. Explain concepts clearly and simply.`;
      }

      // Inject Past History Context
      try {
          const historyData = localStorage.getItem('tutoring_history');
          if (historyData) {
              const history: SavedSession[] = JSON.parse(historyData);
              // Take the last 2 sessions to keep context manageable
              const recentHistory = history.slice(0, 2).map(s => {
                  const date = new Date(s.timestamp).toLocaleDateString();
                  const summary = s.summary 
                      ? `Topic: ${s.summary.overview}, Key Points: ${s.summary.knowledgePoints.join(', ')}`
                      : `Content Preview: ${s.preview}`;
                  return `  - [${date}]: ${summary}`;
              }).join('\n');

              if (recentHistory) {
                  currentSystemInstruction += `\n\n- **Past Learning Context**: Here is a summary of the student's recent learning history. USE THIS to make connections (e.g., "Remember when we learned about [Topic] last time? This is similar...").\n${recentHistory}`;
              }
          }
      } catch (e) {
          console.error("Failed to load history for context", e);
      }

      // Inject Exam Database Context
      if (examDatabase.length > 0) {
          currentSystemInstruction += `\n\n- **Student's Past Exams Database**: The student has uploaded the following past exams. If you recognize a question from the video stream that matches these exams, YOU MUST tell the student when they took this exam (the date) and what exam it is. \n`;
          // Limit to top 3 exams to prevent token limit issues
          examDatabase.slice(0, 3).forEach(exam => {
              currentSystemInstruction += `  - Exam Name: ${exam.name}, Date: ${exam.date}, Content Snippet: ${exam.content.substring(0, 300)}...\n`;
          });
      }

      currentSystemInstruction += `\n\n- **CRITICAL LANGUAGE RULE**: You MUST use ONLY Chinese (全中文) for all subjects and interactions, EXCEPT when the subject is explicitly English. Do NOT use any English words, phrases, or translations unless you are teaching an English lesson.`;

      // Inject AI Response Speed
      if (aiResponseSpeed === 'slow') {
          currentSystemInstruction += `\n\n- **Response Speed (SLOW)**: The student prefers a slower pace. Speak slowly, be less verbose, give very small hints, and wait longer for the student to think before offering more help.`;
      } else if (aiResponseSpeed === 'fast') {
          currentSystemInstruction += `\n\n- **Response Speed (FAST)**: The student prefers a faster pace. Speak quickly, be more direct, and provide more comprehensive hints or explanations immediately.`;
      } else {
          currentSystemInstruction += `\n\n- **Response Speed (NORMAL)**: Provide hints and explanations at a normal, balanced pace.`;
      }

      // 5. Connect to Gemini Live
      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: { role: 'system', parts: [{ text: currentSystemInstruction }] },
          tools: [{
              functionDeclarations: [
                  {
                      name: "generateDiagram",
                      description: "当学生遇到几何题、物理题、应用题等需要画图辅助理解的题目时，调用此函数生成教学示意图。必须传入题目的完整描述。",
                      parameters: {
                          type: Type.OBJECT,
                          properties: {
                              questionContent: {
                                  type: Type.STRING,
                                  description: "题目的完整内容或核心条件描述"
                              }
                          },
                          required: ["questionContent"]
                      }
                  },
                  {
                      name: "reportUnclearVideo",
                      description: "当检测到学生拍摄的画面模糊、反光、被遮挡或无法看清时，调用此函数在界面上显示友好的提示，引导学生调整摄像头或光线。",
                      parameters: {
                          type: Type.OBJECT,
                          properties: {
                              reason: {
                                  type: Type.STRING,
                                  description: "画面不清晰的原因，例如：'画面模糊'、'反光严重'、'手指遮挡'等"
                              }
                          },
                          required: ["reason"]
                      }
                  },
                  {
                      name: "triggerVariantQuestion",
                      description: "当讲解完一个知识点后，为了验证学生是否掌握（进行费曼互动测试），或者当学生同意或想做一道变式练习题时，调用此函数在界面上生成美观的互动型“变式练习题”交互弹窗。不可直接给原题答案，而是出一道和原题思路高度类似、数值不同的变式题考考学生并交互解答。",
                      parameters: {
                          type: Type.OBJECT,
                          properties: {
                              title: {
                                  type: Type.STRING,
                                  description: "变式练习题的简洁标题，例如：'等边三角形面积变式'、'动量守恒变式'等"
                              },
                              question: {
                                  type: Type.STRING,
                                  description: "变式练习题的完整题目描述，难度和知识点保持一致"
                              },
                              options: {
                                  type: Type.ARRAY,
                                  items: {
                                      type: Type.STRING
                                  },
                                  description: "选择题选项。尽量提供标准的4个选项列表，如果是大题/填空题可传入空列表。例如：['A. 5°', 'B. 10°', 'C. 15°', 'D. 20°']"
                              },
                              correctAnswer: {
                                  type: Type.STRING,
                                  description: "变式题的最佳/正确答案。若是选择题，传入对应大写字母例如 'A'。若非选择题，传入简要正确字符串或数字。"
                              },
                              explanation: {
                                  type: Type.STRING,
                                  description: "详细步骤和友好解析（用于给学生演示或解答）"
                              }
                          },
                          required: ["title", "question", "options", "correctAnswer", "explanation"]
                      }
                  }
              ]
          }], // Enable the tool
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          speechConfig: { 
            voiceConfig: { 
                prebuiltVoiceConfig: { 
                    voiceName: ['Puck', 'Charon', 'Kore', 'Fenrir', 'Zephyr'].includes(userProfile.voiceName || '') ? userProfile.voiceName : 'Kore' 
                } 
            } 
          },
        },
        callbacks: {
          onopen: () => {
            console.log('Gemini Live Connection Opened');
            setConnectionState(ConnectionState.CONNECTED);
            connectionStateRef.current = ConnectionState.CONNECTED;
            
            if (stream && stream.getVideoTracks().length > 0) {
                startVideoStreaming(sessionPromise);
            }

            if (!inputContextRef.current) {
                console.warn("No audio context available for live session.");
                return;
            }

            if (!stream || stream.getAudioTracks().length === 0) {
                console.log("No audio stream available for input. Using oscillator for dummy audio.");
                
                const oscillator = inputContextRef.current.createOscillator();
                oscillatorRef.current = oscillator;
                const gainNode = inputContextRef.current.createGain();
                gainNode.gain.value = 0; // Silent
                oscillator.connect(gainNode);
                
                const analyser = inputContextRef.current.createAnalyser();
                analyser.fftSize = 64;
                analyser.smoothingTimeConstant = 0.5;
                gainNode.connect(analyser);
                setInputAnalyser(analyser);

                const processor = inputContextRef.current.createScriptProcessor(4096, 1, 1);
                processorRef.current = processor;
                const currentSampleRate = inputContextRef.current.sampleRate || PCM_SAMPLE_RATE;
                processor.onaudioprocess = (e) => {
                    if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcmBlob = createPcmBlob(inputData, currentSampleRate);
                    sessionPromise.then(session => {
                        if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                        try {
                            session.sendRealtimeInput({ audio: pcmBlob });
                        } catch(e) { console.error("Inner error sending dummy audio:", e) }
                    }).catch(e => console.error("Error sending dummy audio:", e));
                };
                
                gainNode.connect(processor);
                processor.connect(inputContextRef.current.destination);
                oscillator.start();
                
                return;
            }
            const source = inputContextRef.current.createMediaStreamSource(stream);
            
            const analyser = inputContextRef.current.createAnalyser();
            analyser.fftSize = 64;
            analyser.smoothingTimeConstant = 0.5;
            source.connect(analyser);
            setInputAnalyser(analyser);

            const processor = inputContextRef.current.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;
            const currentSampleRate = inputContextRef.current.sampleRate || PCM_SAMPLE_RATE;
            processor.onaudioprocess = (e) => {
              let inputData;
              if (isMicMutedRef.current) {
                  // Send dummy audio when muted to prevent server tokenizer crash
                  inputData = new Float32Array(4096);
              } else {
                  inputData = e.inputBuffer.getChannelData(0);
              }
              const pcmBlob = createPcmBlob(inputData, currentSampleRate);
              sessionPromise.then(session => {
                  if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                  try {
                      session.sendRealtimeInput({ audio: pcmBlob });
                  } catch(e) { console.error("Inner error sending audio:", e) }
              }).catch(e => console.error("Error sending audio promise:", e));
            };
            source.connect(processor);
            processor.connect(inputContextRef.current.destination);
          },
          onmessage: async (msg: LiveServerMessage) => {
            if (msg.serverContent?.inputTranscription) {
              const text = msg.serverContent.inputTranscription.text;
              if (text) {
                  updateTranscript('user', text, false);
                  
                  // Keyword detection for visual context
                  const keywords = ['帮我', '看看', '解决', '不懂', '难点', '解释', '为什么', '错哪', '题', 'question', 'help', 'look', 'see', 'what', 'wrong'];
                  if (keywords.some(k => text.toLowerCase().includes(k))) {
                      console.log("Visual context trigger detected via text:", text);
                      if (sessionPromiseRef.current) {
                          triggerVisualContext(sessionPromiseRef.current);
                      }
                  }
              }
            }
            if (msg.serverContent?.outputTranscription) {
              const text = msg.serverContent.outputTranscription.text;
              if (text) updateTranscript('model', text, false);
            }
            if (msg.serverContent?.turnComplete) setIsBotSpeaking(false);

            // Handle Tool Calls
            if (msg.toolCall) {
                const functionCalls = msg.toolCall.functionCalls;
                if (functionCalls) {
                    const responses = functionCalls.map(call => {
                        if (call.name === 'generateDiagram') {
                            const args = call.args as { questionContent: string };
                            handleGenerateDiagram(args.questionContent);
                            return {
                                id: call.id,
                                name: call.name,
                                response: { result: "Diagram generation started." }
                            };
                        }
                        if (call.name === 'reportUnclearVideo') {
                            const args = call.args as { reason: string };
                            handleReportUnclearVideo(args.reason);
                            return {
                                id: call.id,
                                name: call.name,
                                response: { result: "Warning displayed to user." }
                            };
                        }
                        if (call.name === 'triggerVariantQuestion') {
                            const args = call.args as unknown as VariantQuestion;
                            handleTriggerVariantQuestion(args);
                            return {
                                id: call.id,
                                name: call.name,
                                response: { result: "Variant question displayed in interface for interactive solving." }
                            };
                        }
                        return {
                            id: call.id,
                            name: call.name,
                            response: { result: "Function not implemented" }
                        };
                    });
                    
                    sessionPromise.then(session => {
                        if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                        try {
                            session.sendToolResponse({ functionResponses: responses });
                        } catch(e) { console.error("Inner error tool response:", e); }
                    }).catch(e => console.error("Tool response error:", e));
                }
            }

            const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && outputContextRef.current && outputNodeRef.current) {
              setIsBotSpeaking(true);
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputContextRef.current.currentTime);
              const audioBuffer = await decodeAudioData(decode(audioData), outputContextRef.current, OUTPUT_SAMPLE_RATE, 1);
              const source = outputContextRef.current.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputNodeRef.current);
              source.addEventListener('ended', () => {
                activeSourcesRef.current.delete(source);
                if (activeSourcesRef.current.size === 0) setIsBotSpeaking(false);
              });
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              activeSourcesRef.current.add(source);
            }
            if (msg.serverContent?.interrupted) {
                console.log("Interrupted");
                activeSourcesRef.current.forEach(s => s.stop());
                activeSourcesRef.current.clear();
                nextStartTimeRef.current = 0;
                setIsBotSpeaking(false);
            }
          },
          onclose: () => {
            console.log('Connection Closed');
            if (silenceTimerRef.current) {
                clearInterval(silenceTimerRef.current);
                silenceTimerRef.current = null;
            }
            stopSession();
          },
          onerror: (err) => {
            console.error('Gemini Error:', err);
            setError(err instanceof Error ? err.message : "连接发生错误，请重试。");
            if (silenceTimerRef.current) {
                clearInterval(silenceTimerRef.current);
                silenceTimerRef.current = null;
            }
            stopSession();
          }
        }
      });
      sessionPromise.catch(err => {
          console.error('Async session promise failed', err);
          setError(err instanceof Error ? err.message : String(err));
          setConnectionState(ConnectionState.ERROR);
          connectionStateRef.current = ConnectionState.ERROR;
          stopSession();
      });
      sessionPromiseRef.current = sessionPromise;

    } catch (err) {
      console.error("Session start error:", err);
      // We only get here if Gemini connection fails, since media errors are now caught
      setError(err instanceof Error ? err.message : "无法启动会话");
      setConnectionState(ConnectionState.ERROR);
      stopSession();
    }
  };

  const handleUploadExamToDatabase = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) {
        alert("文件大小不能超过 10MB");
        return;
    }

    setIsUploadingExam(true);
    try {
        const isPdf = file.type === 'application/pdf' || file.name.endsWith('.pdf');
        let extractedText = '';

        if (isPdf) {
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const textContent = await page.getTextContent();
                const pageText = textContent.items.map((item: any) => item.str).join(' ');
                extractedText += `Page ${i}:\n${pageText}\n\n`;
            }
        } else if (file.type.startsWith('text/') || file.name.endsWith('.txt')) {
            extractedText = await file.text();
        } else {
            alert("目前仅支持上传 PDF 或文本文件作为题库。");
            setIsUploadingExam(false);
            return;
        }

        const newRecord: ExamRecord = {
            id: `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
            name: file.name.replace(/\.[^/.]+$/, ""), // Remove extension
            date: new Date().toISOString().split('T')[0], // Default to today
            content: extractedText,
            timestamp: Date.now()
        };

        saveExamDatabase([newRecord, ...examDatabase]);
        
    } catch (error) {
        console.error("Error uploading exam:", error);
        alert("解析文件失败，请重试。");
    } finally {
        setIsUploadingExam(false);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    }
  };

  const handleSendFile = useCallback(async (file: File) => {
    if (!sessionPromiseRef.current) return;
    if (file.size > 5 * 1024 * 1024) {
        alert("文件大小不能超过 5MB");
        return;
    }
    try {
        const isImage = file.type.startsWith('image/');
        const isText = file.type.startsWith('text/') || file.name.endsWith('.txt') || file.name.endsWith('.md') || file.name.endsWith('.json');
        const isPdf = file.type === 'application/pdf' || file.name.endsWith('.pdf');

        if (isPdf) {
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            let fullText = '';
            
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const textContent = await page.getTextContent();
                const pageText = textContent.items.map((item: any) => item.str).join(' ');
                fullText += `Page ${i}:\n${pageText}\n\n`;
            }

            const messageText = `[用户上传了PDF文件: ${file.name}]\n${fullText}`;
            const displayText = `[用户上传了PDF文件: ${file.name}]`;
            triggerAITutor(`用户上传了PDF文件: ${file.name}\n内容: ${fullText}`);
            return;
        }

        if (isText) {
             const text = await file.text();
             const fullText = `[用户上传了文本文件: ${file.name}]\n${text}`;
             const displayText = `[用户上传了文本文件: ${file.name}]`;
             triggerAITutor(`用户上传了文本文件: ${file.name}\n内容: ${text}`);
             return;
        }

        if (isImage) {
            const base64 = await blobToBase64(file);
            const mimeType = file.type;
            const msgText = `[用户上传了图片: ${file.name}]`;
            updateTranscript('user', msgText, true);
            
            // Run OCR on the uploaded image using state-of-the-art Gemini 2.5 Flash
            try {
                const text = await performHighPrecisionOcr(file);
                if (text) {
                    const ocrText = `[系统：通过高精度AI OCR识别到上传图片中的文字内容如下]\n${text}`;
                    console.log("High precision OCR result on upload:", text);
                    
                    triggerAITutor(`通过高精度AI识别并同步到当前图片中的文字内容如下: ${text}`);
                    
                    const classification = await classifyQuestion(text);
                    if (classification) {
                        triggerAITutor(`题型分析完成。学科：${classification.subject}，知识点：${classification.topic}，题型：${classification.questionType}，难度：${classification.difficulty}，核心概念：${classification.keyConcepts.join(', ')}。请根据此题型特点进行针对性讲解。`);
                    }
                }
            } catch (e) {
                console.error("High-precision OCR Error on uploaded image:", e);
            }

            sessionPromiseRef.current.then(session => {
                 if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                 try {
                     session.sendRealtimeInput({ video: { mimeType, data: base64 } });
                 } catch(e) { console.error(e); }
                 
                 // Trigger response
                 if (typeof session.sendClientContent === 'function') {
                     setTimeout(() => {
                         if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                         try {
                             session.sendClientContent({
                                  turns: [{ role: 'user', parts: [{ text: `我上传了一张图片 (${file.name})，请帮我看看。` }] }],
                                  turnComplete: true
                             });
                         } catch(e) { console.error(e); }
                     }, 200);
                 }
            }).catch(e => console.error("Error sending file:", e));
            return;
        }

        alert("目前仅支持图片和文本文件");
    } catch (e) {
        console.error("File upload failed", e);
        alert("文件处理失败");
    }
  }, [triggerAITutor, updateTranscript]);

  const handleAskExplain = (arg1?: any, arg2?: string) => {
      let t: 'knowledge' | 'eye' | undefined;
      let c: string | undefined;

      if (typeof arg1 === 'string') {
          t = arg1 as 'knowledge' | 'eye';
          c = arg2;
      } else {
          t = activePopup?.type;
          c = activePopup?.content;
      }
      
      if (!t || !c) return;

      const prompt = `请详细为我讲解一下这个${t === 'knowledge' ? '知识点' : '题眼'}：${c}`;
      triggerAITutor(`学生请求详细讲解${t === 'knowledge' ? '知识点' : '题眼'}：${c}`);
      setActivePopup(null);
  };

  const performBackgroundOCR = async () => {
      if (!videoRef.current || !canvasRef.current || isProcessingOCR) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx || video.readyState < 2) return;

      setIsProcessingOCR(true);
      try {
          // Capture current frame
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          
          // Apply image enhancement
          applyImageEnhancement(canvas);
          
          const dataUrl = canvas.toDataURL('image/jpeg');
          
          // Run OCR
          const worker = await getOcrWorker();
          if (!worker) {
              setIsProcessingOCR(false);
              return;
          }
          const result = await worker.recognize(dataUrl);
          
          const text = result.data.text.trim();
          if (text && text.length > 10) {
              // Similarity check to avoid spamming the same text
              if (lastOcrTextRef.current) {
                  const prev = lastOcrTextRef.current;
                  const curr = text;
                  
                  // Calculate bigram Sørensen–Dice coefficient for a more robust similarity check
                  const getBigrams = (str: string) => {
                      const bigrams = new Set<string>();
                      const cleanStr = str.replace(/\s+/g, '');
                      for (let i = 0; i < cleanStr.length - 1; i++) {
                          bigrams.add(cleanStr.substring(i, i + 2));
                      }
                      return bigrams;
                  };

                  const prevBigrams = getBigrams(prev);
                  const currBigrams = getBigrams(curr);
                  
                  let similarity = 0;
                  if (prevBigrams.size > 0 && currBigrams.size > 0) {
                      let intersection = 0;
                      for (const bg of currBigrams) {
                          if (prevBigrams.has(bg)) intersection++;
                      }
                      similarity = (2.0 * intersection) / (prevBigrams.size + currBigrams.size);
                  }

                  if (similarity > 0.65 || prev === curr || prev.includes(curr) || curr.includes(prev)) {
                      // Text is similar. Student is NOT writing (or writing very little).
                      // Do not update lastInteractionTimeRef.
                      return; 
                  }
              }
              
              // Text is significantly different. Student IS writing.
              lastOcrTextRef.current = text;
              lastInteractionTimeRef.current = Date.now(); // Reset silence timer
              silenceLevelRef.current = 0; // Reset silence level
          }
      } catch (e) {
          console.error("Background OCR Error:", e);
      } finally {
          setIsProcessingOCR(false);
      }
  };

  useEffect(() => {
      if (connectionState === ConnectionState.CONNECTED) {
          // Run background OCR every 15 seconds to save CPU cycles
          ocrIntervalRef.current = setInterval(performBackgroundOCR, 15000);
      } else {
          if (ocrIntervalRef.current) {
              clearInterval(ocrIntervalRef.current);
          }
      }
      return () => {
          if (ocrIntervalRef.current) {
              clearInterval(ocrIntervalRef.current);
          }
      };
  }, [connectionState]);

  return (
    <div className="flex h-screen w-full bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white overflow-hidden">
      {/* CSS Animations */}
      <style>{`
        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        @keyframes scan {
            0% { top: 0%; opacity: 0; }
            15% { opacity: 1; }
            85% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }
        .scan-line {
            position: absolute;
            left: 0;
            width: 100%;
            height: 2px;
            background: rgba(99, 102, 241, 0.8);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.8);
            animation: scan 3s linear infinite;
            pointer-events: none;
            z-index: 5;
        }
        @keyframes ripple {
            0% { transform: scale(1); opacity: 0.6; }
            100% { transform: scale(2.5); opacity: 0; }
        }
        .ripple-effect::before, .ripple-effect::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: inherit;
            border-radius: inherit;
            z-index: -1;
            animation: ripple 2s cubic-bezier(0, 0.2, 0.8, 1) infinite;
        }
        .ripple-effect::after {
            animation-delay: 1s;
        }
        @keyframes bracket-in {
             0% { transform: scale(1.2); opacity: 0; }
             100% { transform: scale(1); opacity: 1; }
        }
        .bracket-anim {
             animation: bracket-in 0.3s ease-out forwards;
        }
      `}</style>

      {/* Main Video Area */}
      <div className="flex-1 flex flex-col relative">
        {/* Header - Only show when connected */}
        {connectionState !== ConnectionState.DISCONNECTED && (
            <div className="absolute top-0 left-0 right-0 z-10 p-4 bg-gradient-to-b from-black/70 to-transparent flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <div className="bg-indigo-600 p-2 rounded-lg">
                        <Video size={20} className="text-gray-900 dark:text-white" />
                    </div>
                    <h1 className="text-xl font-bold tracking-tight">Live Tutor</h1>
                </div>
                
                <div className="flex items-center gap-4">
                     <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-black/40 backdrop-blur-md border border-white/10">
                        <span className={`w-2 h-2 rounded-full ${
                            connectionState === ConnectionState.CONNECTED ? 'bg-green-500 animate-pulse' : 
                            connectionState === ConnectionState.CONNECTING ? 'bg-yellow-500' : 'bg-red-500'
                        }`}></span>
                        <span className="text-xs font-medium text-gray-600 dark:text-gray-300">
                            {connectionState === ConnectionState.CONNECTED ? '实时连接中' : 
                             connectionState === ConnectionState.CONNECTING ? '连接中...' : '未连接'}
                        </span>
                     </div>
                </div>
            </div>
        )}

        {/* Video Feed */}
        <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden group">
            <video 
                ref={videoRef} 
                className={`w-full h-full object-cover transition-transform duration-300 ${isVideoMirrored ? 'scale-x-[-1]' : 'scale-x-100'}`}
                playsInline 
                muted 
            />
            {/* Hidden canvas for frame extraction */}
            <canvas ref={canvasRef} className="hidden" />
            
            {/* Media Warning Overlay */}
            {mediaWarning && connectionState === ConnectionState.CONNECTED && (
                <div className="absolute inset-0 flex items-center justify-center z-10 bg-gray-900/80 backdrop-blur-sm">
                    <div className="bg-gray-800 border border-yellow-500/30 p-6 rounded-2xl max-w-md text-center shadow-2xl">
                        <div className="w-16 h-16 bg-yellow-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Camera className="text-yellow-500" size={32} />
                        </div>
                        <h3 className="text-xl font-bold text-white mb-2">无摄像头画面</h3>
                        <p className="text-gray-300 text-sm leading-relaxed">{mediaWarning}</p>
                    </div>
                </div>
            )}

            {/* Quality Warning Overlay */}
            {qualityWarning && connectionState === ConnectionState.CONNECTED && !mediaWarning && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 animate-in slide-in-from-top-4 fade-in duration-300">
                    <div className="bg-yellow-500/90 backdrop-blur-md text-white px-6 py-3 rounded-full shadow-lg flex items-center gap-3 border border-yellow-400">
                        <AlertCircle size={20} className="animate-pulse" />
                        <span className="font-medium text-sm whitespace-nowrap">{qualityWarning}</span>
                    </div>
                </div>
            )}

            {/* Persistent Scanner Overlay */}
            {connectionState === ConnectionState.CONNECTED && !mediaWarning && (
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10 overflow-hidden">
                    <div className="relative w-[70%] h-[50%] border-2 border-white/80 rounded shadow-[0_0_0_9999px_rgba(0,0,0,0.5)]">
                        <div className="absolute -top-[2px] -left-[2px] w-5 h-5 border-t-4 border-l-4 border-emerald-500"></div>
                        <div className="absolute -top-[2px] -right-[2px] w-5 h-5 border-t-4 border-r-4 border-emerald-500"></div>
                        <div className="absolute -bottom-[2px] -left-[2px] w-5 h-5 border-b-4 border-l-4 border-emerald-500"></div>
                        <div className="absolute -bottom-[2px] -right-[2px] w-5 h-5 border-b-4 border-r-4 border-emerald-500"></div>
                        {isVisualContextActive && (
                            <div className="absolute left-0 right-0 h-[2px] bg-emerald-400/80 shadow-[0_0_15px_rgba(52,211,153,0.8)] animate-[scan_2s_linear_infinite] top-0"></div>
                        )}
                    </div>
                    <p className="text-white mt-5 text-sm drop-shadow-md z-20 font-medium">
                        请将题目放入框内，老师帮你看看
                    </p>
                </div>
            )}

            {/* Scanner / Focus Overlay */}
            {scannerActive && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20">
                     <div className="relative w-3/5 h-2/5 bracket-anim">
                         {/* Corners */}
                         <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-cyan-400 rounded-tl-lg shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
                         <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-cyan-400 rounded-tr-lg shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
                         <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-cyan-400 rounded-bl-lg shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
                         <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-cyan-400 rounded-br-lg shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
                         
                         {/* Scanning Line */}
                         <div className="absolute left-0 right-0 h-[2px] bg-cyan-400/80 shadow-[0_0_15px_rgba(34,211,238,0.8)] animate-[scan_2s_linear_infinite] top-0"></div>
                         
                         {/* Background tint */}
                         <div className="absolute inset-0 bg-cyan-400/5"></div>
                         
                         {/* Label */}
                         <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-cyan-950/80 border border-cyan-500/50 px-3 py-1 rounded-full flex items-center gap-2">
                             <Target size={14} className="text-cyan-400 animate-pulse" />
                             <span className="text-cyan-100 text-xs font-bold tracking-wider">AI 正在识别重点区域</span>
                         </div>
                     </div>
                </div>
            )}

            {/* Blur Warning Overlay */}
            {showBlurWarning && (
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 animate-in fade-in zoom-in duration-300 pointer-events-none">
                     <div className="bg-black/60 backdrop-blur-md border border-yellow-500/50 text-yellow-100 px-8 py-6 rounded-2xl shadow-2xl flex flex-col items-center gap-4">
                         <div className="p-3 bg-yellow-500/20 rounded-full animate-pulse">
                            <ScanEye size={40} className="text-yellow-400" />
                         </div>
                         <div className="text-center">
                            <h3 className="font-bold text-xl mb-1 text-gray-900 dark:text-white">画面有点模糊</h3>
                            <p className="text-sm text-yellow-200/90">请拿稳手机或调整距离，让我看清楚一点</p>
                         </div>
                     </div>
                </div>
            )}

            {/* Insight Card Overlay (Left side) */}
            {(insightData.knowledge || insightData.eye) && (
                <div className="absolute top-24 left-6 max-w-[280px] z-20 flex flex-col gap-3 animate-in fade-in slide-in-from-left-8 duration-700">
                    {insightData.knowledge && (
                        <div 
                            onClick={() => setActivePopup({ type: 'knowledge', content: insightData.knowledge! })}
                            className="bg-white dark:bg-gray-900/80 backdrop-blur-lg border border-indigo-500/40 p-4 rounded-2xl shadow-2xl border-l-4 border-l-indigo-500 transform transition-all hover:scale-105 cursor-pointer hover:bg-gray-100 dark:bg-gray-800/90 group"
                        >
                            <div className="flex items-center gap-2 mb-2 text-indigo-300 font-bold text-sm tracking-wide group-hover:text-indigo-200">
                                <Lightbulb size={18} className="fill-indigo-500/20" />
                                <span>核心知识点</span>
                            </div>
                            <p className="text-gray-800 dark:text-gray-100 text-sm leading-relaxed font-medium line-clamp-2">{insightData.knowledge}</p>
                            <div 
                                className="text-xs text-indigo-400 mt-2 flex items-center transition-opacity hover:text-indigo-300 hover:underline"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    handleAskExplain('knowledge', insightData.knowledge!);
                                }}
                            >
                                <Sparkles size={12} className="mr-1" /> 点击让 AI 详细讲解
                            </div>
                        </div>
                    )}
                    {insightData.eye && (
                        <div 
                            onClick={() => setActivePopup({ type: 'eye', content: insightData.eye! })}
                            className="bg-white dark:bg-gray-900/80 backdrop-blur-lg border border-emerald-500/40 p-4 rounded-2xl shadow-2xl border-l-4 border-l-emerald-500 transform transition-all hover:scale-105 cursor-pointer hover:bg-gray-100 dark:bg-gray-800/90 group" 
                            style={{animationDelay: '150ms'}}
                        >
                            <div className="flex items-center gap-2 mb-2 text-emerald-300 font-bold text-sm tracking-wide group-hover:text-emerald-200">
                                <Key size={18} className="fill-emerald-500/20" />
                                <span>解题关键 (题眼)</span>
                            </div>
                            <p className="text-gray-800 dark:text-gray-100 text-sm leading-relaxed font-medium line-clamp-2">{insightData.eye}</p>
                            <div 
                                className="text-xs text-emerald-400 mt-2 flex items-center transition-opacity hover:text-emerald-300 hover:underline"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    handleAskExplain('eye', insightData.eye!);
                                }}
                            >
                                <Sparkles size={12} className="mr-1" /> 点击让 AI 详细讲解
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Insight Detail Popup */}
            {activePopup && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-in fade-in duration-200">
                    <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
                        {/* Popup Header */}
                        <div className={`p-4 flex justify-between items-center ${activePopup.type === 'knowledge' ? 'bg-indigo-900/30' : 'bg-emerald-900/30'} border-b border-white/5`}>
                             <div className={`flex items-center gap-2 font-bold ${activePopup.type === 'knowledge' ? 'text-indigo-300' : 'text-emerald-300'}`}>
                                {activePopup.type === 'knowledge' ? <Lightbulb size={20} /> : <Key size={20} />}
                                <span>{activePopup.type === 'knowledge' ? '核心知识点' : '解题关键'}</span>
                             </div>
                             <button onClick={() => setActivePopup(null)} className="p-1 hover:bg-white/10 rounded-full text-gray-400 dark:text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:text-white transition-colors">
                                <X size={20} />
                             </button>
                        </div>
                        
                        {/* Popup Content */}
                        <div className="p-6">
                            <p className="text-lg text-gray-800 dark:text-gray-100 leading-relaxed font-medium">{activePopup.content}</p>
                        </div>

                        {/* Popup Footer (Actions) */}
                        <div className="p-4 bg-black/20 flex gap-3">
                             <button 
                                onClick={handleAskExplain}
                                className="flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-gray-900 dark:text-white font-semibold transition-all shadow-lg shadow-indigo-600/20 active:scale-95"
                             >
                                <MessageCircleQuestion size={18} />
                                让 AI 详细讲解
                             </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Diagram Modal */}
            {(diagramSvg || isGeneratingDiagram) && (
                <div className="absolute inset-0 z-40 flex items-center justify-center p-4 pointer-events-none">
                    <div className="bg-white border border-gray-200 w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300 flex flex-col pointer-events-auto">
                        <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                            <h3 className="font-bold text-lg text-gray-900 flex items-center gap-2">
                                <Sparkles size={20} className="text-indigo-500" />
                                教学示意图
                            </h3>
                            <button 
                                onClick={() => setDiagramSvg(null)} 
                                className="p-1 hover:bg-gray-200 rounded-full text-gray-400 hover:text-gray-900 transition-colors"
                            >
                                <X size={20} />
                            </button>
                        </div>
                        <div className="p-6 flex items-center justify-center bg-white min-h-[400px]">
                            {isGeneratingDiagram ? (
                                <div className="flex flex-col items-center gap-4">
                                    <Loader2 size={40} className="animate-spin text-indigo-500" />
                                    <p className="text-gray-500 animate-pulse">正在生成示意图...</p>
                                </div>
                            ) : diagramSvg ? (
                                <div 
                                    className="w-full h-full flex items-center justify-center"
                                    dangerouslySetInnerHTML={{ __html: diagramSvg }} 
                                />
                            ) : null}
                        </div>
                    </div>
                </div>
            )}

            {/* AI Smart High-Precision OCR Scanner Modal */}
            <AnimatePresence>
              {showOcrModal && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-6 bg-black/85 backdrop-blur-md">
                  <motion.div 
                    key="ocr-modal"
                    initial={{ opacity: 0, scale: 0.9, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 15 }}
                    transition={{ type: 'spring', damping: 25, stiffness: 350 }}
                    className="bg-white dark:bg-gray-950 border border-emerald-500/30 w-full max-w-4xl rounded-[2.5rem] shadow-2xl overflow-hidden flex flex-col max-h-[85vh] relative"
                    id="high-precision-ocr-modal"
                  >
                     {/* Header */}
                     <div className="px-8 py-6 bg-gradient-to-r from-emerald-600 to-indigo-900 flex justify-between items-center text-white shrink-0">
                         <div className="flex items-center gap-4">
                             <div className="bg-white/15 p-2.5 rounded-2xl">
                                 <ScanEye size={28} className="text-emerald-200" />
                             </div>
                             <div>
                                 <span className="text-xs uppercase tracking-widest font-bold text-emerald-200 block">高精拍照识字</span>
                                 <h3 className="font-extrabold text-2xl text-white mt-0.5">AI高精字迹识别与同步</h3>
                             </div>
                         </div>
                         <button 
                             onClick={() => setShowOcrModal(false)}
                             className="p-2 hover:bg-white/10 rounded-full text-emerald-200 hover:text-white transition-colors cursor-pointer"
                             id="close-ocr-btn"
                         >
                             <X size={26} />
                         </button>
                     </div>

                     {/* Content: Two Columns */}
                     <div className="p-8 overflow-y-auto flex-1 grid grid-cols-1 md:grid-cols-2 gap-8 bg-gray-50 dark:bg-gray-950">
                          {/* Left Column: Captured Photo Preview */}
                          <div className="flex flex-col gap-4">
                              <span className="text-sm font-bold text-gray-500 dark:text-gray-400 flex items-center gap-1.5">
                                  <Camera size={16} /> 拍照存根
                              </span>
                              <div className="relative flex-1 min-h-[300px] max-h-[450px] rounded-3xl overflow-hidden border border-gray-200 dark:border-gray-800 bg-black flex items-center justify-center group shadow-md flex-1">
                                  {ocrCapturedImage ? (
                                      <img 
                                          src={ocrCapturedImage} 
                                          alt="Snapshot" 
                                          className="max-w-full max-h-full object-contain"
                                          referrerPolicy="no-referrer"
                                      />
                                  ) : (
                                      <div className="text-gray-400 flex flex-col items-center gap-2">
                                          <Loader2 className="animate-spin text-emerald-500" size={32} />
                                          <p className="text-sm">尚未截获画面</p>
                                      </div>
                                  )}
                                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm">
                                      <p className="text-white text-xs font-semibold uppercase bg-black/60 px-4 py-2 rounded-full border border-white/10">当前抓包画质合格</p>
                                  </div>
                              </div>
                          </div>

                          {/* Right Column: OCR Text Result */}
                          <div className="flex flex-col gap-4">
                              <span className="text-sm font-bold text-gray-500 dark:text-gray-400 flex items-center gap-1.5">
                                  <FileText size={16} /> 识别转义文字
                              </span>
                              
                              <div className="flex-1 min-h-[300px] flex flex-col rounded-3xl border border-gray-200 dark:border-gray-850 bg-white dark:bg-gray-900 overflow-hidden shadow-md flex-1">
                                  {isOcrLoading ? (
                                      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center gap-4">
                                          <div className="relative">
                                              <Loader2 size={56} className="animate-spin text-emerald-500" />
                                              <Sparkles size={20} className="text-indigo-400 absolute -top-1 -right-1 animate-bounce" />
                                          </div>
                                          <div className="space-y-1">
                                              <h4 className="font-bold text-lg text-gray-800 dark:text-white">高精度 AI 正在辨识中</h4>
                                              <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed max-w-xs">
                                                  正在对你的手写笔迹、公式及表格符号进行深层分析，请稍等片刻...
                                              </p>
                                          </div>
                                      </div>
                                  ) : ocrTextResult ? (
                                      <div className="flex-1 flex flex-col p-6 space-y-4">
                                          <textarea 
                                              value={ocrTextResult}
                                              onChange={(e) => setOcrTextResult(e.target.value)}
                                              className="flex-1 w-full bg-transparent border-0 focus:ring-0 resize-none font-mono text-base text-gray-800 dark:text-gray-100 leading-relaxed focus:outline-none scrollbar-thin overflow-y-auto"
                                              placeholder="在这里轻微修改识别的字迹，使其100%符合实体笔记内容"
                                          />
                                          
                                          {isOcrSynced && (
                                              <div className="p-3.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl text-sm flex items-center gap-2">
                                                  <Check size={16} className="shrink-0" />
                                                  <span>已同步给AI导师！导师已准确获取并在实时解答中参照此段笔记。</span>
                                              </div>
                                          )}
                                      </div>
                                  ) : (
                                      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center text-gray-500">
                                          <FileText size={36} className="mb-2 text-gray-400" />
                                          <p className="text-sm">无字迹识别数据</p>
                                      </div>
                                  )}
                              </div>
                          </div>
                      </div>

                      {/* Footer Controls */}
                      <div className="p-6 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-850/80 flex justify-between items-center gap-4 shrink-0">
                          <div>
                              <button
                                  disabled={isOcrLoading}
                                  onClick={async () => {
                                      if (!videoRef.current || !canvasRef.current) return;
                                      setIsOcrLoading(true);
                                      setOcrTextResult(null);
                                      setIsOcrSynced(false);
                                      
                                      const video = videoRef.current;
                                      const canvas = canvasRef.current;
                                      const ctx = canvas.getContext('2d');
                                      if (ctx && video.readyState >= 2) {
                                          canvas.width = video.videoWidth;
                                          canvas.height = video.videoHeight;
                                          ctx.drawImage(video, 0, 0);
                                          
                                          // Apply image enhancement
                                          applyImageEnhancement(canvas);
                                          
                                          const dataUrl = canvas.toDataURL('image/jpeg');
                                          setOcrCapturedImage(dataUrl);
                                          
                                          try {
                                              const textResult = await performHighPrecisionOcr(dataUrl);
                                              setOcrTextResult(textResult);
                                          } catch (err) {
                                              console.error("Retake snapshot OCR failed:", err);
                                              setOcrTextResult("识别失败，请重试。");
                                          } finally {
                                              setIsOcrLoading(false);
                                          }
                                      }
                                  }}
                                  className="px-6 py-3 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-750 text-gray-800 dark:text-gray-200 text-base font-bold rounded-2xl transition-all flex items-center gap-2 disabled:opacity-50 cursor-pointer"
                              >
                                  <RefreshCw size={18} className={isOcrLoading ? 'animate-spin' : ''} />
                                  <span>重新拍照</span>
                              </button>
                          </div>
                          
                          <div className="flex items-center gap-3">
                              <button
                                  disabled={!ocrTextResult || isOcrLoading}
                                  onClick={() => {
                                      if (ocrTextResult) {
                                          navigator.clipboard.writeText(ocrTextResult);
                                          alert("文字内容已成功复制到剪贴板！");
                                      }
                                  }}
                                  className="px-6 py-3 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-750 text-gray-800 dark:text-gray-200 text-base font-bold rounded-2xl transition-all disabled:opacity-50 cursor-pointer"
                              >
                                  <span>复制字迹</span>
                              </button>
                              
                              <button
                                  disabled={!ocrTextResult || isOcrLoading || isOcrSynced || connectionState !== ConnectionState.CONNECTED}
                                  onClick={() => {
                                      if (ocrTextResult) {
                                          handleSendMessage(
                                              `[系统高精度图片识字] 这是我通过高精拍照识别到的最新手写笔记文字，请参照此内容为我作答、点拨 and 指导，请绝对知悉：\n${ocrTextResult}`,
                                              "[高精拍照识字] 已上传最新笔记文字",
                                              false
                                          );
                                          setIsOcrSynced(true);
                                      }
                                  }}
                                  className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white text-base font-bold rounded-2xl shadow-lg shadow-emerald-600/20 active:scale-[0.98] transition-all disabled:opacity-50 disabled:bg-emerald-600/70 cursor-pointer"
                              >
                                  <span>同步给AI导师</span>
                              </button>
                          </div>
                      </div>
                   </motion.div>
                 </div>
               )}
             </AnimatePresence>

            {/* Feynman Variant Question Modal */}
            <AnimatePresence>
              {activeVariantQuestion && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-6 bg-black/70 backdrop-blur-md">
                  <motion.div 
                    key="variant-modal"
                    initial={{ opacity: 0, scale: 0.9, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 15 }}
                    transition={{ type: 'spring', damping: 25, stiffness: 350 }}
                    className="bg-white dark:bg-gray-900 border border-indigo-500/30 w-full max-w-2xl rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]"
                    id="feynman-variant-modal"
                  >
                     {/* Title Header */}
                     <div className="p-6 bg-gradient-to-r from-indigo-600 to-indigo-900 flex justify-between items-center text-white shrink-0">
                         <div className="flex items-center gap-3">
                             <div className="bg-white/15 p-2.5 rounded-xl">
                                 <Target size={26} className="text-indigo-200 animate-pulse" />
                             </div>
                             <div>
                                 <span className="text-xs uppercase tracking-widest font-semibold text-indigo-200 block">费曼互动特训</span>
                                 <h3 className="font-bold text-2xl text-white mt-0.5">{activeVariantQuestion.title}</h3>
                             </div>
                         </div>
                         <button 
                             onClick={() => setActiveVariantQuestion(null)}
                             className="p-1.5 hover:bg-white/10 rounded-full text-indigo-200 hover:text-white transition-colors animate-pulse"
                             id="close-variant-btn"
                         >
                             <X size={24} />
                         </button>
                     </div>

                     {/* Content Area */}
                     <div className="p-8 overflow-y-auto flex-1 space-y-6">
                          {/* Question Text */}
                          <div className="bg-indigo-50/50 dark:bg-indigo-950/20 p-6 rounded-2xl border border-indigo-100 dark:border-indigo-900/40">
                              <span className="text-xs font-semibold px-2.5 py-1 bg-indigo-100 dark:bg-indigo-900/60 text-indigo-700 dark:text-indigo-300 rounded-lg inline-block mb-3">变式练习</span>
                              <p className="text-xl text-gray-800 dark:text-gray-100 font-medium leading-relaxed" id="variant-question-text">
                                  {activeVariantQuestion.question}
                              </p>
                          </div>

                          {/* Options Interface */}
                          {activeVariantQuestion.options && activeVariantQuestion.options.length > 0 ? (
                              <div className="grid grid-cols-1 gap-3.5" id="variant-options-list">
                                  {activeVariantQuestion.options.map((option, idx) => {
                                      const optionLetter = option.trim().charAt(0).toUpperCase();
                                      const isSelected = selectedVariantAnswer === optionLetter;
                                      
                                      let bgClass = "bg-gray-50 hover:bg-gray-100 dark:bg-gray-800/40 dark:hover:bg-gray-800 border-gray-200 dark:border-gray-800";
                                      if (isSelected) {
                                          bgClass = "bg-indigo-50 dark:bg-indigo-950/40 border-indigo-500 ring-2 ring-indigo-500/20";
                                      }
                                      if (showVariantFeedback) {
                                          const isCorrectOption = optionLetter === activeVariantQuestion.correctAnswer.trim().toUpperCase();
                                          if (isCorrectOption) {
                                              bgClass = "bg-emerald-50 dark:bg-emerald-950/40 border-emerald-500 ring-2 ring-emerald-500/30 text-emerald-950 dark:text-emerald-100";
                                          } else if (isSelected) {
                                              bgClass = "bg-rose-50 dark:bg-rose-950/40 border-rose-500 ring-2 ring-rose-500/30 text-rose-950 dark:text-rose-100";
                                          }
                                      }

                                      return (
                                          <button
                                              key={idx}
                                              id={`option-btn-${optionLetter}`}
                                              disabled={showVariantFeedback}
                                              onClick={() => setSelectedVariantAnswer(optionLetter)}
                                              className={`p-5 rounded-2xl border text-left text-lg font-medium transition-all duration-200 flex items-center justify-between ${bgClass}`}
                                          >
                                              <span className="text-gray-800 dark:text-gray-200 leading-relaxed">{option}</span>
                                              <div className="flex items-center gap-2">
                                                  {showVariantFeedback && optionLetter === activeVariantQuestion.correctAnswer.trim().toUpperCase() && (
                                                      <div className="bg-emerald-500 text-white rounded-full p-1 shadow-md">
                                                          <Check size={16} strokeWidth={3} />
                                                      </div>
                                                  )}
                                                  {!showVariantFeedback && (
                                                      <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all ${isSelected ? 'border-indigo-600 bg-indigo-600 text-white' : 'border-gray-300 dark:border-gray-600'}`}>
                                                          {isSelected && <div className="w-2.5 h-2.5 bg-white rounded-full" />}
                                                      </div>
                                                  )}
                                              </div>
                                          </button>
                                      );
                                  })}
                              </div>
                          ) : (
                              // Open Text Answer format
                              <div className="space-y-3" id="variant-text-answer-block">
                                  <label className="text-sm font-semibold text-gray-500 dark:text-gray-400">请输入你的解答答案：</label>
                                  <input 
                                      type="text"
                                      disabled={showVariantFeedback}
                                      value={variantTextAnswer}
                                      onChange={(e) => setVariantTextAnswer(e.target.value)}
                                      placeholder="例如数字、最终角、简答..."
                                      className="w-full p-5 rounded-2xl border border-gray-200 dark:border-gray-800 text-lg bg-gray-50 dark:bg-gray-800/40 focus:ring-4 focus:ring-indigo-500/15 focus:outline-none focus:border-indigo-500 transition-all text-gray-900 dark:text-white"
                                      id="variant-text-input"
                                  />
                              </div>
                          )}

                          {/* Feedback Animation Area */}
                          {showVariantFeedback && (
                              <motion.div
                                  initial={{ opacity: 0, y: -10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  className={`p-6 rounded-2xl border ${isVariantAnswerCorrect ? 'bg-emerald-50/70 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-900/40 text-emerald-950 dark:text-emerald-100' : 'bg-rose-50/70 dark:bg-rose-950/20 border-rose-200 dark:border-rose-900/40 text-rose-950 dark:text-rose-100'}`}
                                  id="variant-feedback-box"
                              >
                                  <div className="flex items-start gap-3.5">
                                      <div className={`p-2.5 rounded-xl ${isVariantAnswerCorrect ? 'bg-emerald-500 text-white' : 'bg-rose-500 text-white'}`}>
                                          {isVariantAnswerCorrect ? <Sparkles size={22} className="animate-bounce" /> : <Lightbulb size={22} />}
                                      </div>
                                      <div className="space-y-1.5 flex-1">
                                          <h4 className="font-bold text-xl">
                                              {isVariantAnswerCorrect ? '太棒了！完全正确 🎉' : '没关系，老师给你思路线索 💡'}
                                          </h4>
                                          <p className="text-base leading-relaxed text-gray-700 dark:text-gray-300">
                                              {isVariantAnswerCorrect 
                                                ? `你成功通过了这道变式测试！太有创意的思维了，对知识点已经融会贯通，真棒！` 
                                                : `别气馁，让我们一起了解一下解题的奥妙！`}
                                          </p>
                                      </div>
                                  </div>

                                  {/* Explanation Details */}
                                  <div className="mt-5 pt-5 border-t border-gray-200/50 dark:border-gray-800/50 space-y-2">
                                      <span className="text-xs font-semibold px-2 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded">详细解析</span>
                                      <p className="text-base text-gray-600 dark:text-gray-300 leading-relaxed font-sans font-normal antialiased">
                                          {activeVariantQuestion.explanation}
                                      </p>
                                  </div>
                              </motion.div>
                          )}
                     </div>

                     {/* Footer Controls */}
                     <div className="p-6 bg-gray-50 dark:bg-gray-950/40 border-t border-gray-100 dark:border-gray-800 flex justify-end gap-3.5 shrink-0">
                         {!showVariantFeedback ? (
                             <button
                                 id="submit-variant-btn"
                                 disabled={activeVariantQuestion.options && activeVariantQuestion.options.length > 0 ? !selectedVariantAnswer : !variantTextAnswer.trim()}
                                 onClick={() => {
                                     let correct = false;
                                     if (activeVariantQuestion.options && activeVariantQuestion.options.length > 0) {
                                         correct = selectedVariantAnswer?.trim().toUpperCase() === activeVariantQuestion.correctAnswer.trim().toUpperCase();
                                     } else {
                                         const ansNormalized = variantTextAnswer.trim().toLowerCase();
                                         const correctNormalized = activeVariantQuestion.correctAnswer.trim().toLowerCase();
                                         correct = ansNormalized.includes(correctNormalized) || correctNormalized.includes(ansNormalized);
                                     }
                                     setIsVariantAnswerCorrect(correct);
                                     setShowVariantFeedback(true);
                                 }}
                                 className="px-8 py-3.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:bg-indigo-600 disabled:hover:bg-indigo-600 cursor-pointer disabled:cursor-not-allowed text-white text-lg font-bold rounded-2xl shadow-lg shadow-indigo-600/20 active:scale-[0.98] transition-all flex items-center gap-2"
                             >
                                 <span>提交解答</span>
                                 <ArrowRight size={18} />
                             </button>
                         ) : (
                             <button
                                 id="next-variant-btn"
                                 onClick={() => {
                                     setActiveVariantQuestion(null);
                                 }}
                                 className="px-8 py-3.5 bg-gradient-to-r from-emerald-600 to-indigo-600 hover:from-emerald-500 hover:to-indigo-500 text-white text-lg font-bold rounded-2xl shadow-lg transition-all active:scale-[0.98]"
                             >
                                 继续探索
                             </button>
                         )}
                     </div>
                  </motion.div>
                </div>
              )}
            </AnimatePresence>

            {/* Session Summary Modal - Temporarily Hidden */}
            {false && showSummaryModal && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-in fade-in duration-300">
                    <div className="bg-white dark:bg-gray-900 border border-indigo-500/30 w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300 flex flex-col max-h-[80vh]">
                        {/* Header */}
                        <div className="p-5 border-b border-gray-300 dark:border-gray-700/50 bg-gradient-to-r from-indigo-900/40 to-gray-900 flex justify-between items-center">
                            <div className="flex items-center gap-3">
                                <div className="bg-indigo-600 p-2 rounded-lg shadow-lg shadow-indigo-600/20">
                                    <BookOpen size={24} className="text-gray-900 dark:text-white" />
                                </div>
                                <div>
                                    <h3 className="font-bold text-xl text-gray-900 dark:text-white">今日学习报告</h3>
                                    <p className="text-xs text-indigo-300">AI 智能生成的辅导总结</p>
                                </div>
                            </div>
                            <button 
                                onClick={() => setShowSummaryModal(false)} 
                                className="p-2 hover:bg-white/10 rounded-full text-gray-400 dark:text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:text-white transition-colors"
                                disabled={isGeneratingSummary}
                            >
                                <X size={20} />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6">
                            {isGeneratingSummary ? (
                                <div className="flex flex-col items-center justify-center h-48 gap-4">
                                    <Loader2 size={40} className="animate-spin text-indigo-500" />
                                    <p className="text-gray-400 dark:text-gray-500 dark:text-gray-400 animate-pulse">正在整理学习笔记，请稍候...</p>
                                </div>
                            ) : sessionSummary ? (
                                <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                                    {/* Overview Section */}
                                    <div className="bg-gray-100 dark:bg-gray-800/50 rounded-xl p-4 border border-gray-300 dark:border-gray-700">
                                        <h4 className="text-indigo-400 text-sm font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
                                            <FileText size={16} /> 学习概览
                                        </h4>
                                        <p className="text-gray-700 dark:text-gray-200 leading-relaxed text-sm">
                                            {sessionSummary.overview}
                                        </p>
                                    </div>

                                    {/* Knowledge Points Section */}
                                    <div>
                                        <h4 className="text-emerald-400 text-sm font-bold uppercase tracking-wider mb-3 flex items-center gap-2">
                                            <Lightbulb size={16} /> 核心知识点
                                        </h4>
                                        <ul className="space-y-2">
                                            {sessionSummary.knowledgePoints.map((point, idx) => (
                                                <li key={idx} className="flex flex-col gap-2 bg-gray-100 dark:bg-gray-800/30 p-3 rounded-lg border border-gray-300 dark:border-gray-700/50 hover:border-emerald-500/30 transition-colors">
                                                    <div className="flex gap-3 items-start">
                                                        <span className="flex-shrink-0 w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xs font-bold mt-0.5">
                                                            {idx + 1}
                                                        </span>
                                                        <span className="text-gray-600 dark:text-gray-300 text-sm flex-1">{point}</span>
                                                    </div>
                                                    <div className="pl-8">
                                                        <button 
                                                            onClick={() => {
                                                                handleAskExplain('knowledge', point);
                                                                setShowSummaryModal(false);
                                                            }}
                                                            className="text-xs flex items-center gap-1 text-indigo-500 hover:text-indigo-400 font-medium transition-colors py-1 px-2 rounded hover:bg-indigo-500/10"
                                                        >
                                                            <Sparkles size={12} />
                                                            AI 详细讲解
                                                        </button>
                                                    </div>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                    
                                    <div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg text-xs text-blue-300 text-center">
                                        这份报告已保存到历史记录中，随时可以在左侧回顾。
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center text-gray-400 dark:text-gray-500 py-10">
                                    <p>暂无总结内容</p>
                                </div>
                            )}
                        </div>

                        {/* Footer */}
                        <div className="p-4 bg-gray-100 dark:bg-gray-800/50 border-t border-gray-300 dark:border-gray-700/50 flex justify-end">
                            <button 
                                onClick={() => setShowSummaryModal(false)}
                                disabled={isGeneratingSummary}
                                className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 text-gray-900 dark:text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                完成
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Settings & Profile Overlay (Top Right inside video) - Only show when connected */}
            {connectionState !== ConnectionState.DISCONNECTED && (
                <div className="absolute top-20 right-4 flex flex-col gap-3 z-30">
                    {/* Handwritten AI OCR Scan Toggle */}
                    <button 
                        onClick={async () => {
                            if (!videoRef.current || !canvasRef.current) return;
                            const video = videoRef.current;
                            const canvas = canvasRef.current;
                            const ctx = canvas.getContext('2d');
                            if (!ctx || video.readyState < 2) return;
                            
                            // Capture high-resolution photo from the live stream
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            ctx.drawImage(video, 0, 0);
                            
                            // Apply image enhancement
                            applyImageEnhancement(canvas);
                            
                            const dataUrl = canvas.toDataURL('image/jpeg');
                            setOcrCapturedImage(dataUrl);
                            setOcrTextResult(null);
                            setIsOcrLoading(true);
                            setIsOcrSynced(false);
                            setShowOcrModal(true);
                            
                            try {
                                const textResult = await performHighPrecisionOcr(dataUrl);
                                setOcrTextResult(textResult);
                            } catch (err) {
                                console.error("Snapshot OCR failed:", err);
                                setOcrTextResult("识别失败，请确保摄像头画面清晰且对准笔记，然后重试。");
                            } finally {
                                setIsOcrLoading(false);
                            }
                        }}
                        className="p-3 rounded-full bg-black/50 hover:bg-black/70 text-emerald-400 dark:text-emerald-300 backdrop-blur-md border border-white/10 transition-all hover:scale-110 active:scale-95 flex items-center justify-center cursor-pointer group shadow-lg"
                        title="AI 拍照高精度识字"
                        id="ai-ocr-scan-btn"
                    >
                        <ScanEye size={20} className="group-hover:animate-pulse" />
                    </button>

                    {/* Mirror Toggle */}
                    <button 
                        onClick={() => setIsVideoMirrored(!isVideoMirrored)}
                        className={`p-3 rounded-full text-gray-900 dark:text-white backdrop-blur-md border border-white/10 transition-all ${isVideoMirrored ? 'bg-indigo-600/80 hover:bg-indigo-600' : 'bg-black/50 hover:bg-black/70'}`}
                        title={isVideoMirrored ? "关闭镜像" : "开启镜像"}
                    >
                        <FlipHorizontal size={20} />
                    </button>

                    {/* Profile Toggle */}
                    <button 
                        onClick={() => {
                            setShowProfileModal(true);
                            setShowSettings(false);
                        }}
                        className="p-3 rounded-full bg-black/50 hover:bg-black/70 text-gray-900 dark:text-white backdrop-blur-md border border-white/10 transition-all group relative"
                        title="个人资料"
                    >
                        <UserRoundPen size={20} />
                        {userProfile.avatar && (
                            <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-indigo-500 text-xs shadow-sm ring-2 ring-black">
                                {userProfile.avatar}
                            </span>
                        )}
                    </button>

                    {/* Camera Settings Toggle */}
                    <button 
                        onClick={() => {
                            setShowSettings(!showSettings);
                            setShowProfileModal(false);
                        }}
                        className="p-3 rounded-full bg-black/50 hover:bg-black/70 text-gray-900 dark:text-white backdrop-blur-md border border-white/10 transition-all"
                        title="设置"
                    >
                        <Settings size={20} />
                    </button>
                    
                    {/* Settings Dropdown */}
                    {showSettings && (
                        <div className="flex flex-col gap-2 p-3 rounded-2xl bg-black/60 backdrop-blur-xl border border-white/10 animate-in fade-in zoom-in duration-200 w-64 shadow-2xl">
                            <div className="text-xs font-semibold text-gray-400 dark:text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wider">摄像头设置</div>
                            
                            {/* Device Selector */}
                            <div className="relative mb-2">
                                <select 
                                    value={selectedCameraId}
                                    onChange={(e) => switchCamera(e.target.value)}
                                    className="w-full bg-gray-100 dark:bg-gray-800/80 text-gray-900 dark:text-white text-sm rounded-lg p-2 pl-8 outline-none border border-gray-300 dark:border-gray-700 hover:border-indigo-500 appearance-none"
                                >
                                    {videoDevices.map(device => (
                                        <option key={device.deviceId} value={device.deviceId}>
                                            {device.label || `摄像头 ${device.deviceId.slice(0, 5)}...`}
                                        </option>
                                    ))}
                                </select>
                                <Camera size={14} className="absolute left-2.5 top-2.5 text-gray-400 dark:text-gray-500 dark:text-gray-400 pointer-events-none" />
                            </div>

                            {/* Smart Quality Toggle */}
                            <div className="mb-2">
                                 <button 
                                    onClick={() => setIsAutoQuality(!isAutoQuality)}
                                    className={`w-full flex items-center justify-between p-2 rounded-lg text-sm transition-colors ${isAutoQuality ? 'bg-emerald-600/30 text-emerald-200' : 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:bg-gray-700'}`}
                                 >
                                     <div className="flex items-center gap-2">
                                         <Wifi size={16} />
                                         <span>智能画质调节</span>
                                     </div>
                                     <div className={`w-8 h-4 rounded-full relative transition-colors ${isAutoQuality ? 'bg-emerald-500' : 'bg-gray-300 dark:bg-gray-600'}`}>
                                         <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${isAutoQuality ? 'left-4.5' : 'left-0.5'}`} style={{left: isAutoQuality ? '18px' : '2px'}} />
                                     </div>
                                 </button>
                                 {isAutoQuality && (
                                    <div className="mt-1 px-2 flex justify-between items-center text-[10px]">
                                        <span className="text-gray-400 dark:text-gray-500">当前网络:</span>
                                        <span className={`font-medium ${
                                            networkStatus === 'good' ? 'text-green-400' : 
                                            networkStatus === 'moderate' ? 'text-yellow-400' : 
                                            networkStatus === 'poor' ? 'text-red-400' : 'text-gray-400 dark:text-gray-500 dark:text-gray-400'
                                        }`}>
                                            {networkStatus === 'good' ? '极佳' : networkStatus === 'moderate' ? '一般' : networkStatus === 'poor' ? '较差' : '未知'}
                                        </span>
                                    </div>
                                 )}
                            </div>

                            {/* Frame Rate / Speed Slider */}
                            <div className={`mb-3 px-1 transition-opacity ${isAutoQuality ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
                                <div className="flex items-center justify-between text-xs text-gray-400 dark:text-gray-500 dark:text-gray-400 mb-2">
                                    <div className="flex items-center gap-1">
                                        <Gauge size={14} />
                                        <span>{isAutoQuality ? "识别速度 (自动)" : "识别速度 (手动)"}</span>
                                    </div>
                                    <span className="text-indigo-400 font-mono">{videoFrameRate} FPS</span>
                                </div>
                                <input 
                                    type="range" 
                                    min="0.5" 
                                    max="5" 
                                    step="0.5" 
                                    value={videoFrameRate}
                                    onChange={(e) => setVideoFrameRate(parseFloat(e.target.value))}
                                    className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all"
                                />
                                <div className="flex justify-between text-[10px] text-gray-600 mt-1 font-medium">
                                    <span>省流</span>
                                    <span>极速</span>
                                </div>
                            </div>

                            <div className="h-px bg-gray-200 dark:bg-gray-700/50 my-2" />

                            {/* AI Response Speed */}
                            <div className="mb-3 px-1">
                                <div className="flex items-center justify-between text-xs text-gray-400 dark:text-gray-500 mb-2">
                                    <div className="flex items-center gap-1">
                                        <Sparkles size={14} />
                                        <span>AI 回复速度</span>
                                    </div>
                                    <span className="text-indigo-400 font-mono">
                                        {aiResponseSpeed === 'slow' ? '较慢' : aiResponseSpeed === 'fast' ? '较快' : '正常'}
                                    </span>
                                </div>
                                <input 
                                    type="range" 
                                    min="0" 
                                    max="2" 
                                    step="1" 
                                    value={aiResponseSpeed === 'slow' ? 0 : aiResponseSpeed === 'fast' ? 2 : 1}
                                    onChange={(e) => {
                                        const val = parseInt(e.target.value);
                                        setAiResponseSpeed(val === 0 ? 'slow' : val === 2 ? 'fast' : 'normal');
                                    }}
                                    className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all"
                                />
                                <div className="flex justify-between text-[10px] text-gray-600 mt-1 font-medium">
                                    <span>较慢</span>
                                    <span>正常</span>
                                    <span>较快</span>
                                </div>
                            </div>

                            <div className="h-px bg-gray-200 dark:bg-gray-700/50 my-2" />

                            {/* Image Enhancement Pipeline */}
                            <div className="mb-3 px-1">
                                <div className="flex items-center gap-1.5 text-xs text-gray-400 dark:text-gray-500 mb-2">
                                    <Sun size={14} className="text-emerald-400" />
                                    <span className="font-semibold text-emerald-400">纸张字迹硬核增强</span>
                                </div>
                                <div className="grid grid-cols-3 gap-1 mb-2">
                                    {[
                                        { id: 'none', label: '原图' },
                                        { id: 'document', label: '纸张白化' },
                                        { id: 'contrast', label: '高对比' },
                                        { id: 'grayscale', label: '黑白' },
                                        { id: 'custom', label: '自定义' }
                                    ].map(p => (
                                        <button
                                            key={p.id}
                                            onClick={() => setImageEnhancePreset(p.id as any)}
                                            className={`py-1 px-1 rounded text-[10px] font-bold border transition-all ${
                                                imageEnhancePreset === p.id 
                                                    ? 'bg-emerald-600/30 border-emerald-500/50 text-emerald-300 shadow-sm'
                                                    : 'bg-gray-100/5 dark:bg-gray-800/40 border-transparent text-gray-400 hover:text-white hover:bg-gray-800/70'
                                            }`}
                                        >
                                            {p.label}
                                        </button>
                                    ))}
                                </div>

                                <div className="mt-2.5 mb-1 space-y-1.5 p-1.5 bg-neutral-900/40 rounded-xl border border-white/5">
                                    <button
                                        onClick={() => setEnableAdaptiveThreshold(!enableAdaptiveThreshold)}
                                        className={`w-full flex items-center justify-between p-1 rounded-lg text-[10px] transition-colors ${
                                            enableAdaptiveThreshold 
                                                ? 'bg-emerald-600/20 text-emerald-300' 
                                                : 'text-gray-400 hover:bg-gray-800'
                                        }`}
                                    >
                                        <span className="font-semibold">自适应局部降噪 (去阴影)</span>
                                        <div className={`w-6 h-3 rounded-full relative transition-colors ${enableAdaptiveThreshold ? 'bg-emerald-500' : 'bg-gray-600'}`}>
                                            <div className="absolute top-0.5 w-2 h-2 bg-white rounded-full transition-all" style={{ left: enableAdaptiveThreshold ? '14px' : '2px' }} />
                                        </div>
                                    </button>

                                    <button
                                        onClick={() => setEnableDeskewing(!enableDeskewing)}
                                        className={`w-full flex items-center justify-between p-1 rounded-lg text-[10px] transition-colors ${
                                            enableDeskewing 
                                                ? 'bg-emerald-600/20 text-emerald-300' 
                                                : 'text-gray-400 hover:bg-gray-800'
                                        }`}
                                    >
                                        <span className="font-semibold">倾斜字迹自动校正</span>
                                        <div className={`w-6 h-3 rounded-full relative transition-colors ${enableDeskewing ? 'bg-emerald-500' : 'bg-gray-600'}`}>
                                            <div className="absolute top-0.5 w-2 h-2 bg-white rounded-full transition-all" style={{ left: enableDeskewing ? '14px' : '2px' }} />
                                        </div>
                                    </button>
                                </div>

                                {imageEnhancePreset === 'custom' && (
                                    <div className="space-y-2 mt-2 bg-black/40 p-2 rounded-xl border border-white/5 animate-in fade-in slide-in-from-top-1 duration-150">
                                        <div>
                                            <div className="flex justify-between text-[9px] text-gray-400 mb-0.5">
                                                <span>对比度增强</span>
                                                <span className="text-emerald-400 font-mono">+{contrastLevel}%</span>
                                            </div>
                                            <input 
                                                type="range"
                                                min="0"
                                                max="100"
                                                value={contrastLevel}
                                                onChange={(e) => setContrastLevel(parseInt(e.target.value))}
                                                className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-emerald-500"
                                            />
                                        </div>
                                        <div>
                                            <div className="flex justify-between text-[9px] text-gray-400 mb-0.5">
                                                <span>亮度修正</span>
                                                <span className="text-emerald-400 font-mono">{brightnessLevel >= 0 ? `+${brightnessLevel}` : brightnessLevel}%</span>
                                            </div>
                                            <input 
                                                type="range"
                                                min="-50"
                                                max="50"
                                                value={brightnessLevel}
                                                onChange={(e) => setBrightnessLevel(parseInt(e.target.value))}
                                                className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-emerald-500"
                                            />
                                        </div>
                                        <div>
                                            <div className="flex justify-between text-[9px] text-gray-400 mb-0.5">
                                                <span>文字锐化</span>
                                                <span className="text-emerald-400 font-mono">+{sharpenLevel}%</span>
                                            </div>
                                            <input 
                                                type="range"
                                                min="0"
                                                max="100"
                                                value={sharpenLevel}
                                                onChange={(e) => setSharpenLevel(parseInt(e.target.value))}
                                                className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-emerald-500"
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="h-px bg-gray-200 dark:bg-gray-700/50 my-2" />

                            {/* Mirror Toggle */}
                            <button 
                                onClick={() => setIsVideoMirrored(!isVideoMirrored)}
                                className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors ${isVideoMirrored ? 'bg-indigo-600/30 text-indigo-200' : 'hover:bg-gray-200 dark:bg-gray-700/50'}`}
                            >
                                <div className="flex items-center gap-2">
                                    <RefreshCw size={16} />
                                    <span>镜像画面</span>
                                </div>
                                <div className={`w-8 h-4 rounded-full relative transition-colors ${isVideoMirrored ? 'bg-indigo-500' : 'bg-gray-300 dark:bg-gray-600'}`}>
                                    <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${isVideoMirrored ? 'left-4.5' : 'left-0.5'}`} style={{left: isVideoMirrored ? '18px' : '2px'}} />
                                </div>
                            </button>
                            
                            {/* Speaker Mute Toggle */}
                            <button 
                                onClick={() => setIsSpeakerMuted(!isSpeakerMuted)}
                                className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors ${isSpeakerMuted ? 'bg-red-500/20 text-red-200' : 'hover:bg-gray-200 dark:bg-gray-700/50'}`}
                            >
                                <div className="flex items-center gap-2">
                                    {isSpeakerMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                                    <span>AI 语音播放</span>
                                </div>
                                <div className={`w-8 h-4 rounded-full relative transition-colors ${!isSpeakerMuted ? 'bg-green-500' : 'bg-gray-300 dark:bg-gray-600'}`}>
                                    <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all`} style={{left: !isSpeakerMuted ? '18px' : '2px'}} />
                                </div>
                            </button>
                        </div>
                    )}
                </div>
            )}

            {/* Smart Terminal Hardware Physical Controls Overlay */}
            {connectionState === ConnectionState.CONNECTED && (
                <div id="hardware-panel" className="absolute right-4 bottom-24 z-30 flex flex-col gap-3 py-3 px-3 rounded-2xl bg-gray-950/85 md:bg-gray-950/90 border border-slate-700/50 shadow-2xl transition-all duration-300 w-44 pointer-events-auto select-none">
                    {/* Hardware Bezel Title and Indicator */}
                    <div className="flex flex-col items-center border-b border-gray-800 pb-2">
                        <div className="flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                            <span className="font-mono text-[9px] uppercase tracking-widest text-slate-300 font-bold">Smart Tutor AI</span>
                        </div>
                        <span className="text-[10px] text-slate-500 mt-0.5">智能硬件物理键</span>
                    </div>
                    
                    {/* Physical Mute Mic Button */}
                    <button 
                        onClick={() => {
                            const next = !isMicMuted;
                            setIsMicMuted(next);
                            playMuteSound(next);
                        }}
                        className={`w-full flex items-center justify-between p-2 rounded-xl border transition-all cursor-pointer ${
                            isMicMuted 
                            ? 'bg-red-950/40 border-red-500/50 text-red-300 hover:bg-red-955/65' 
                            : 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300 hover:bg-emerald-955/65'
                        }`}
                        title="物理静音键 (快捷键: M)"
                    >
                        <div className="flex items-center gap-2">
                            <div className={`w-2.5 h-2.5 rounded-full shadow-lg border border-black/30 flex items-center justify-center ${isMicMuted ? 'bg-red-500 animate-pulse' : 'bg-emerald-400'}`}>
                                <div className="w-1 h-1 bg-white rounded-full opacity-60" />
                            </div>
                            <span className="text-xs font-semibold">麦克风静音</span>
                        </div>
                        <span className="text-[9px] font-mono bg-slate-800/80 text-slate-400 px-1 py-0.5 rounded border border-white/5">M</span>
                    </button>

                    {/* Physical Mute Speaker Button */}
                    <button 
                        onClick={() => {
                            const next = !isSpeakerMuted;
                            setIsSpeakerMuted(next);
                            playMuteSound(next);
                        }}
                        className={`w-full flex items-center justify-between p-2 rounded-xl border transition-all cursor-pointer ${
                            isSpeakerMuted 
                            ? 'bg-amber-950/40 border-amber-500/50 text-amber-300 hover:bg-amber-955/65' 
                            : 'bg-slate-900/40 border-slate-700/50 text-slate-300 hover:bg-slate-880/65'
                        }`}
                        title="物理扬声器键 (快捷键: S)"
                    >
                        <div className="flex items-center gap-2">
                            <div className={`w-2.5 h-2.5 rounded-full shadow-lg border border-black/30 flex items-center justify-center ${isSpeakerMuted ? 'bg-amber-500 animate-pulse' : 'bg-sky-400'}`}>
                                <div className="w-1 h-1 bg-white rounded-full opacity-60" />
                            </div>
                            <span className="text-xs font-semibold">音响静音</span>
                        </div>
                        <span className="text-[9px] font-mono bg-slate-800/80 text-slate-400 px-1 py-0.5 rounded border border-white/5">S</span>
                    </button>

                    {/* Simulated circular "Wake-up" button */}
                    <button 
                        onClick={() => {
                            if (isMicMuted) {
                                setIsMicMuted(false);
                                playMuteSound(false);
                            }
                            playWakeBeep();
                            triggerAITutor("按键唤醒");
                        }}
                        className="w-full h-16 rounded-2xl bg-gradient-to-tr from-indigo-700 via-indigo-600 to-violet-600 hover:from-indigo-600 hover:to-violet-500 text-white shadow-lg active:scale-95 transition-all text-center border border-indigo-500/50 cursor-pointer flex flex-col items-center justify-center overflow-hidden group select-none relative"
                        title="物理唤醒键 (快捷键: 空格键)"
                    >
                        {/* Glow effect */}
                        <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                        <Sparkles size={14} className="text-white animate-pulse mb-0.5" />
                        <span className="text-xs font-bold tracking-wider">物理按键唤醒</span>
                        <span className="text-[8px] opacity-75 mt-0.5 font-mono">Space 空格键</span>
                    </button>

                    <div className="h-px bg-slate-850/80 my-0.5" />

                    {/* Voice Wake-up service */}
                    <div className="flex flex-col gap-1.5 p-1 bg-slate-900/50 rounded-xl border border-white/5">
                        <button 
                            onClick={() => setIsVoiceWakeupEnabled(!isVoiceWakeupEnabled)}
                            className="w-full flex items-center justify-between p-1 rounded-lg text-[10px] text-left transition-colors cursor-pointer"
                        >
                            <span className="font-semibold text-slate-300">语音唤醒服务</span>
                            <div className={`w-6 h-3.5 rounded-full relative transition-colors ${isVoiceWakeupEnabled ? 'bg-indigo-500' : 'bg-slate-600'}`}>
                                <div className="absolute top-0.5 w-2.5 h-2.5 bg-white rounded-full transition-all" style={{left: isVoiceWakeupEnabled ? '12px' : '2px'}} />
                            </div>
                        </button>
                        
                        {isVoiceWakeupEnabled && (
                            <div className="flex items-center gap-1.5 px-1 py-0.5 justify-center border-t border-slate-800/80 mt-1 pt-1.5">
                                <span className={`w-1.5 h-1.5 rounded-full ${isWakeWordListening ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`} />
                                <span className="text-[8px] text-slate-400 scale-95 leading-none font-medium">
                                    {isWakeWordListening ? '呼呼"小苏老师"唤醒' : '正在加载引擎...'}
                                </span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Profile Edit Modal */}
            {showProfileModal && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
                    <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 w-full max-w-sm rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200 max-h-[90vh] flex flex-col">
                        <div className="p-4 border-b border-gray-300 dark:border-gray-700/50 flex justify-between items-center bg-gray-100 dark:bg-gray-800/50 flex-shrink-0">
                            <h3 className="font-bold text-lg text-gray-900 dark:text-white flex items-center gap-2">
                                <UserRoundPen size={20} className="text-indigo-400" />
                                个人资料设置
                            </h3>
                            <button onClick={() => setShowProfileModal(false)} className="p-1 hover:bg-white/10 rounded-full text-gray-400 dark:text-gray-500 hover:text-gray-900 dark:text-white transition-colors">
                                <X size={20} />
                            </button>
                        </div>
                        
                        <div className="p-6 space-y-6 overflow-y-auto">
                            {/* Avatar Selection */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 dark:text-gray-500 dark:text-gray-400 mb-2">选择头像</label>
                                <div className="grid grid-cols-5 gap-2">
                                    {AVATAR_OPTIONS.map(avatar => (
                                        <button
                                            key={avatar}
                                            onClick={() => saveProfile({...userProfile, avatar})}
                                            className={`h-10 w-10 flex items-center justify-center text-xl rounded-full transition-all ${
                                                userProfile.avatar === avatar 
                                                ? 'bg-indigo-600 ring-2 ring-indigo-300 ring-offset-2 ring-offset-gray-900' 
                                                : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:bg-gray-700'
                                            }`}
                                        >
                                            {avatar}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Name Input */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 dark:text-gray-500 dark:text-gray-400 mb-1">昵称 / 名字</label>
                                <input
                                    type="text"
                                    value={userProfile.name}
                                    onChange={(e) => saveProfile({...userProfile, name: e.target.value})}
                                    placeholder="比如: 小明"
                                    className="w-full bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2 text-gray-900 dark:text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                                />
                            </div>

                            {/* Age Input */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 dark:text-gray-500 dark:text-gray-400 mb-1">年龄 (岁)</label>
                                <input
                                    type="number"
                                    value={userProfile.age}
                                    onChange={(e) => saveProfile({...userProfile, age: e.target.value})}
                                    placeholder="AI 将根据年龄调整讲解难度"
                                    className="w-full bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2 text-gray-900 dark:text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                                />
                                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">设置年龄后，AI 会使用更适合该年龄段的语言。</p>
                            </div>

                            {/* Voice Selection */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 dark:text-gray-500 dark:text-gray-400 mb-2">选择 AI 语音</label>
                                <div className="grid grid-cols-1 gap-2 max-h-40 overflow-y-auto pr-1">
                                    {VOICE_OPTIONS.map(voice => (
                                        <button
                                            key={voice.id}
                                            onClick={() => saveProfile({...userProfile, voiceName: voice.id})}
                                            className={`flex items-center justify-between px-4 py-3 rounded-xl border transition-all ${
                                                (userProfile.voiceName || 'Kore') === voice.id 
                                                ? 'bg-indigo-600/20 border-indigo-500 text-gray-900 dark:text-white' 
                                                : 'bg-gray-100 dark:bg-gray-800 border-gray-300 dark:border-gray-700 text-gray-400 dark:text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:bg-gray-700 hover:border-gray-400 dark:border-gray-600'
                                            }`}
                                        >
                                            <div className="flex flex-col items-start">
                                                <div className="flex items-center gap-2">
                                                    <span className={`text-sm font-semibold ${(userProfile.voiceName || 'Kore') === voice.id ? 'text-indigo-300' : 'text-gray-600 dark:text-gray-300'}`}>
                                                        {voice.name}
                                                    </span>
                                                    {voice.gender === 'Female' ? <span className="text-xs text-rose-400 bg-rose-900/30 px-1 rounded">女声</span> : <span className="text-xs text-blue-400 bg-blue-900/30 px-1 rounded">男声</span>}
                                                </div>
                                                <span className="text-xs text-gray-400 dark:text-gray-500 mt-1">{voice.desc}</span>
                                            </div>
                                            {(userProfile.voiceName || 'Kore') === voice.id ? (
                                                <div className="flex items-center gap-2 text-indigo-400">
                                                    <AudioLines size={16} className="animate-pulse" />
                                                    <div className="h-2 w-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.8)]"></div>
                                                </div>
                                            ) : (
                                                <Volume2 size={16} className="text-gray-600" />
                                            )}
                                        </button>
                                    ))}
                                </div>
                            </div>

                        </div>

                        <div className="p-4 bg-gray-100 dark:bg-gray-800/30 border-t border-gray-300 dark:border-gray-700/50 flex-shrink-0">
                            <button 
                                onClick={() => setShowProfileModal(false)}
                                className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-gray-900 dark:text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                            >
                                <Check size={16} />
                                保存并关闭
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Exam Database Modal - Temporarily Hidden */}
            {false && showExamModal && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
                    <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200 max-h-[90vh] flex flex-col">
                        <div className="p-4 border-b border-gray-300 dark:border-gray-700/50 flex justify-between items-center bg-gray-100 dark:bg-gray-800/50 flex-shrink-0">
                            <h3 className="font-bold text-lg text-gray-900 dark:text-white flex items-center gap-2">
                                <Database size={20} className="text-indigo-400" />
                                专属题库管理
                            </h3>
                            <button onClick={() => setShowExamModal(false)} className="p-1 hover:bg-white/10 rounded-full text-gray-400 dark:text-gray-500 hover:text-gray-900 dark:text-white transition-colors">
                                <X size={20} />
                            </button>
                        </div>
                        
                        <div className="p-6 space-y-6 overflow-y-auto flex-grow">
                            <div className="flex justify-between items-center">
                                <div>
                                    <h4 className="text-gray-900 dark:text-white font-medium">已上传试卷 ({examDatabase.length})</h4>
                                    <p className="text-sm text-gray-500">上传历史试卷，AI 导师将在讲解时自动识别并关联。</p>
                                </div>
                                
                                <div>
                                    <input 
                                        type="file" 
                                        accept=".pdf,.txt" 
                                        ref={fileInputRef}
                                        onChange={handleUploadExamToDatabase}
                                        className="hidden" 
                                    />
                                    <button 
                                        onClick={() => fileInputRef.current?.click()}
                                        disabled={isUploadingExam}
                                        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {isUploadingExam ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
                                        上传新试卷
                                    </button>
                                </div>
                            </div>

                            {examDatabase.length === 0 ? (
                                <div className="text-center py-12 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl">
                                    <Database size={48} className="mx-auto text-gray-400 dark:text-gray-600 mb-4" />
                                    <p className="text-gray-500 dark:text-gray-400">暂无试卷数据</p>
                                    <p className="text-sm text-gray-400 dark:text-gray-500 mt-2">点击上方按钮上传 PDF 或文本格式的试卷</p>
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    {examDatabase.map(exam => (
                                        <div key={exam.id} className="p-4 bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-xl flex justify-between items-center group">
                                            <div className="flex items-start gap-3">
                                                <FileText className="text-indigo-400 mt-1" size={20} />
                                                <div>
                                                    <h5 className="font-medium text-gray-900 dark:text-white">{exam.name}</h5>
                                                    <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                                                        <span>上传于: {new Date(exam.timestamp).toLocaleDateString()}</span>
                                                        <span>•</span>
                                                        <span>内容长度: {exam.content.length} 字符</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <button 
                                                onClick={() => saveExamDatabase(examDatabase.filter(e => e.id !== exam.id))}
                                                className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                                                title="删除记录"
                                            >
                                                <Trash2 size={18} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Connection Active Overlays */}
            {connectionState === ConnectionState.CONNECTED && (
                <>
                    {/* Visual Scanning Effect */}
                    <div className="absolute inset-0 pointer-events-none opacity-30">
                         <div className="scan-line" />
                    </div>



                    {/* Status Indicator & Visual Trigger */}
                    <div className="absolute bottom-24 left-1/2 transform -translate-x-1/2 flex flex-col items-center gap-4 z-30 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <button
                            onClick={() => {
                                if (!isBotSpeaking && !isVisualContextActive && sessionPromiseRef.current) {
                                    triggerAITutor("学生点击按钮");
                                }
                            }}
                            disabled={isBotSpeaking || isVisualContextActive}
                            className={`
                                relative flex items-center gap-4 px-8 py-4 rounded-full border backdrop-blur-md transition-all duration-500 shadow-xl text-2xl
                                ${isBotSpeaking 
                                    ? 'bg-emerald-500 border-emerald-400 text-gray-900 dark:text-white ripple-effect scale-110 cursor-default' 
                                    : isVisualContextActive
                                        ? 'bg-indigo-600 border-indigo-500 text-white scale-105 ring-4 ring-indigo-500/30 cursor-default'
                                        : 'bg-white dark:bg-gray-900/90 border-indigo-500/30 text-indigo-600 dark:text-indigo-300 hover:bg-indigo-50 dark:hover:bg-gray-800 hover:scale-105 active:scale-95 cursor-pointer'
                                }
                            `}
                        >
                            {isBotSpeaking ? (
                                <>
                                    <Sparkles size={24} className="animate-spin-slow text-yellow-300" />
                                    <span className="font-semibold tracking-wide">正在讲解...</span>
                                    <div className="h-6 w-[80px] flex items-center">
                                        <AudioVisualizer isActive={true} color="white" width={80} height={24} />
                                    </div>
                                </>
                            ) : isVisualContextActive ? (
                                <>
                                    <ScanEye size={24} className="animate-pulse" />
                                    <span className="font-semibold tracking-wide">正在观察题目...</span>
                                    {/* Breathing light effect */}
                                    <div className="absolute inset-0 rounded-full bg-indigo-500/20 animate-[pulse_2s_ease-in-out_infinite] -z-10"></div>
                                </>
                            ) : (
                                <>
                                    <div className="relative">
                                        <Eye size={24} />
                                        <span className="absolute -top-1 -right-1 flex h-3 w-3">
                                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                                          <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
                                        </span>
                                    </div>
                                    <span className="font-bold tracking-wide">让 AI 看题</span>
                                </>
                            )}
                        </button>
                        {!isBotSpeaking && !isVisualContextActive && (
                            <p className="text-white/80 text-lg text-center mt-2 bg-black/40 px-4 py-2 rounded-full backdrop-blur-sm border border-white/10">
                                说 "帮我看看" 或点击按钮上传画面
                            </p>
                        )}
                    </div>
                </>
            )}
            
            {/* Welcome / Disconnected State */}
            {connectionState === ConnectionState.DISCONNECTED && !error && (
                 <div className="absolute inset-0 flex items-center justify-center z-30">
                    {/* Dark overlay with gradient for better text visibility */}
                    <div className="absolute inset-0 bg-gradient-to-b from-black/80 via-black/60 to-black/90 backdrop-blur-[2px]"></div>
                    
                    <div className="relative text-center p-8 max-w-4xl animate-in fade-in zoom-in duration-500 flex flex-col items-center">
                        <div className="mb-12">
                            <h1 className="text-6xl md:text-8xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-500 mb-6 tracking-tight drop-shadow-2xl">
                                未来的学习体验
                            </h1>
                            <p className="text-gray-700 dark:text-gray-200 text-2xl md:text-4xl font-light leading-relaxed drop-shadow-md">
                                实时连接 AI 导师。使用视频进行互动讲解，为您提供个性化的辅导体验。
                            </p>
                        </div>

                        {/* Fingerprint Profile Card */}
                        <div className="mb-10 transform hover:scale-105 transition-transform duration-300">
                            <div className="relative group cursor-pointer" onClick={() => setShowProfileModal(true)}>
                                {/* Glow effect */}
                                <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-[2rem] blur opacity-75 group-hover:opacity-100 transition duration-1000 group-hover:duration-200"></div>
                                
                                <div className="relative px-8 py-6 bg-white dark:bg-gray-900 ring-1 ring-gray-200 dark:ring-gray-800 rounded-[2rem] flex items-center gap-6 shadow-2xl">
                                    <div className="relative">
                                        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-5xl shadow-inner border-4 border-gray-200 dark:border-gray-800">
                                            {userProfile.avatar || '🎓'}
                                        </div>
                                        <div className="absolute -bottom-2 -right-2 bg-green-500 text-gray-900 dark:text-white text-xs font-bold px-2 py-1 rounded-full border-2 border-gray-900 flex items-center gap-1">
                                            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                                            在线
                                        </div>
                                    </div>
                                    
                                    <div className="text-left">
                                        <div className="text-xs text-indigo-400 font-bold uppercase tracking-widest mb-1 flex items-center gap-1">
                                            <ScanEye size={12} /> 学习指纹档案
                                        </div>
                                        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                                            {userProfile.name || '新同学'}
                                        </h2>
                                        <div className="flex items-center gap-3 text-gray-400 dark:text-gray-500 dark:text-gray-400 text-sm">
                                            <span className="bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded text-xs border border-gray-300 dark:border-gray-700">
                                                {userProfile.age ? `${userProfile.age}岁` : '未设置年龄'}
                                            </span>
                                            <span>•</span>
                                            <span className="flex items-center gap-1">
                                                <Volume2 size={12} /> {VOICE_OPTIONS.find(v => v.id === userProfile.voiceName)?.name.split(' ')[0] || '默认语音'}
                                            </span>
                                        </div>
                                    </div>
                                    
                                    <div className="ml-4 pl-6 border-l border-gray-300 dark:border-gray-700 flex flex-col items-center gap-1 text-gray-400 dark:text-gray-500 group-hover:text-indigo-400 transition-colors">
                                        <Settings size={20} />
                                        <span className="text-xs">设置</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="flex gap-4 mb-10">
                            <button 
                                onClick={startSession}
                                className="px-16 py-6 bg-white text-black hover:bg-gray-100 rounded-full font-bold text-3xl transition-all shadow-[0_0_30px_rgba(255,255,255,0.3)] hover:shadow-[0_0_50px_rgba(255,255,255,0.6)] hover:scale-105 active:scale-95 flex items-center gap-3 group"
                            >
                                <Play size={32} fill="currentColor" className="group-hover:translate-x-1 transition-transform" />
                                立即开始上课
                            </button>

                            {/* Exam Database Button - Temporarily Hidden */}
                            {false && <button
                                onClick={() => setShowExamModal(true)}
                                className="px-8 py-5 bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-100 border border-indigo-500/30 rounded-full font-bold text-xl transition-all hover:scale-105 active:scale-95 flex items-center gap-3"
                            >
                                <Database size={24} />
                                专属题库 ({examDatabase.length})
                            </button>}
                        </div>
                    </div>
                 </div>
            )}

            {/* Error State */}
            {error && (() => {
                const isApiKeyError = error.toLowerCase().includes('api key') || 
                                     error.toLowerCase().includes('api_key') || 
                                     error.toLowerCase().includes('1007') || 
                                     error.toLowerCase().includes('apikey') ||
                                     error.toLowerCase().includes('密钥') ||
                                     error.toLowerCase().includes('密匙');
                return (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-40">
                         <div className="text-left p-6 bg-white dark:bg-gray-900 border border-red-900/50 rounded-xl max-w-md shadow-2xl">
                            <div className="text-center mb-4">
                                <AlertCircle size={40} className="text-red-500 mx-auto mb-2" />
                                <h3 className="text-xl font-bold text-red-500">连接未成功</h3>
                            </div>
                            {isApiKeyError ? (
                                <div className="space-y-3 text-sm text-gray-600 dark:text-gray-300">
                                    <p className="font-semibold text-amber-500 text-center">检测到 API 密钥缺失、未生效或无效 (1007/400 错误)</p>
                                    <p>苏格拉底 AI 视频直播课需要您在 Google AI Studio 中配置您个人的真实 `GEMINI_API_KEY`（完全免费创建且享有免费额度），平台默认的临时测试密钥不支持建立 WebSocket 音视频双向流式连接。</p>
                                    <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 space-y-1.5">
                                        <div className="font-semibold text-gray-800 dark:text-gray-200 text-xs uppercase tracking-wider">🛠️ 精准配置指南:</div>
                                        <ol className="list-decimal list-inside space-y-1 text-xs leading-relaxed">
                                            <li>点击 Google AI Studio 界面右上角的 <strong className="text-gray-900 dark:text-white">Settings</strong> (齿轮) 图标。</li>
                                            <li>在弹出的菜单中选择 <strong className="text-gray-900 dark:text-white">Secrets</strong> 栏目。</li>
                                            <li>点击或添加变量：<strong className="font-mono text-indigo-500 dark:text-indigo-400">GEMINI_API_KEY</strong>。</li>
                                            <li>填入您个人的真实 <strong>Gemini API Key</strong>（可在 Google AI Studio 左侧菜单中点击 <strong className="text-indigo-500">Get API key</strong> 免费创建）。</li>
                                            <li>保存配置，并且<strong>刷新浏览器本页面</strong>重新点击“进入苏格拉底课堂”即可开始完美互动！</li>
                                        </ol>
                                    </div>
                                    <p className="text-xs text-gray-500 text-center mt-3">（由于 API 密钥属于系统机密凭证，AI Studio 不提供在页面中直接输入的输入框，请在右上角系统 Secrets 面板中完成配置。安全有保障 🛡️）</p>
                                </div>
                            ) : (
                                <p className="text-gray-600 dark:text-gray-300 mb-6 text-center">{error}</p>
                            )}
                            <div className="flex justify-center mt-6">
                                <button 
                                    onClick={() => setError(null)}
                                    className="px-6 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:bg-gray-600 rounded-lg text-sm font-medium transition-colors cursor-pointer text-gray-900 dark:text-white"
                                >
                                    理解并关闭
                                </button>
                            </div>
                         </div>
                    </div>
                );
            })()}
        </div>

        {/* Controls Bar - Only show when connected */}
        {connectionState !== ConnectionState.DISCONNECTED && (
            <div className="h-24 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 flex items-center justify-center gap-6 px-4 z-20">
                {connectionState === ConnectionState.CONNECTED ? (
                    <>
                        <div className="flex items-center gap-4">
                            {/* Audio Input Meter */}
                            <div className="flex flex-col items-center justify-center gap-1 mr-2 opacity-80">
                                <AudioVisualizer 
                                    analyser={inputAnalyser} 
                                    color={isMicMuted ? "#4b5563" : "#6366f1"} 
                                    width={50} 
                                    height={20} 
                                />
                            </div>

                            <button 
                                onClick={() => setIsMicMuted(!isMicMuted)}
                                className={`p-4 rounded-full transition-all duration-300 ${
                                    isMicMuted 
                                    ? 'bg-red-500/20 text-red-500 hover:bg-red-500/30 ring-2 ring-red-500/50' 
                                    : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white hover:bg-gray-200 dark:bg-gray-700 hover:scale-110 active:scale-95'
                                }`}
                                title={isMicMuted ? "取消静音" : "静音"}
                            >
                                {isMicMuted ? <MicOff size={24} /> : <Mic size={24} />}
                            </button>
                        </div>

                        <div className="flex items-center gap-4">
                            <button 
                                onClick={() => setShowProfileModal(true)}
                                className="p-4 rounded-full text-gray-900 dark:text-white bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:bg-gray-700 hover:scale-110 active:scale-95 transition-all"
                                title="修改个人资料"
                            >
                                <UserRoundPen size={24} />
                            </button>

                            {/* Save Session Button - Temporarily Hidden */}
                            {false && <button 
                                onClick={() => saveSessionToHistory(true)}
                                disabled={messages.length === 0}
                                className={`p-4 rounded-full text-gray-900 dark:text-white transition-all relative group ${
                                    messages.length === 0 
                                    ? 'bg-gray-100 dark:bg-gray-800 opacity-50 cursor-not-allowed' 
                                    : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:bg-gray-700 hover:scale-110 active:scale-95'
                                }`}
                                title="保存当前对话"
                            >
                                <Save size={24} className={showSaveConfirm ? "text-green-500 transition-colors" : ""} />
                                {showSaveConfirm && (
                                    <span className="absolute -top-10 left-1/2 transform -translate-x-1/2 text-xs font-bold bg-green-500/90 text-gray-900 dark:text-white px-3 py-1.5 rounded-lg shadow-lg animate-in fade-in slide-in-from-bottom-2 whitespace-nowrap z-50">
                                        已保存
                                    </span>
                                )}
                            </button>}

                            <button 
                                onClick={stopSession}
                                className="p-4 rounded-full bg-red-600 text-gray-900 dark:text-white hover:bg-red-500 transition-all shadow-lg hover:shadow-red-600/20 hover:scale-110 active:scale-95"
                                title="结束会话"
                            >
                                <Square size={24} fill="currentColor" />
                            </button>
                        </div>
                    </>
                ) : (
                    <div className="text-sm text-gray-400 dark:text-gray-500 italic">
                        {connectionState === ConnectionState.CONNECTING ? "正在建立连接..." : "等待开始..."}
                    </div>
                )}
            </div>
        )}
      </div>



      {/* Sidebar: Transcript */}
      {connectionState !== ConnectionState.DISCONNECTED && (
          <Transcript 
            messages={messages} 
            userProfile={userProfile}
          />
      )}
    </div>
  );
};

export default LiveTutor;