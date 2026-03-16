import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type } from '@google/genai';
import { Video, Mic, MicOff, Play, Square, AlertCircle, Volume2, Sparkles, Eye, Settings, VolumeX, RefreshCw, Camera, FlipHorizontal, Lightbulb, Key, X, MessageCircleQuestion, ArrowRight, ScanEye, Target, UserRoundPen, Check, ChevronRight, Gauge, Save, AudioLines, Wifi, WifiOff, FileText, Loader2, BookOpen, Sun, Moon, Database, Upload, Trash2 } from 'lucide-react';
import { ConnectionState, ChatMessage, SavedSession, UserProfile, SessionSummary, ExamRecord } from '../types';
import { createPcmBlob, decode, decodeAudioData, blobToBase64 } from '../utils/audioUtils';
import AudioVisualizer from './AudioVisualizer';
import Transcript from './Transcript';

import * as pdfjsLib from 'pdfjs-dist';
// @ts-ignore
import pdfWorker from 'pdfjs-dist/build/pdf.worker.mjs?url';

// Set worker source for PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

// Extend Navigator interface for Network Information API (experimental)
interface NetworkInformation extends EventTarget {
  readonly downlink: number;
  readonly effectiveType: 'slow-2g' | '2g' | '3g' | '4g';
  readonly rtt: number;
  readonly saveData: boolean;
  onchange: EventListener;
}

// Configuration constants
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';

// Voice Options for User Selection
const VOICE_OPTIONS = [
  { id: 'Kore', name: '温柔老师 (Kore)', desc: '舒缓、平和的女性声音', gender: 'Female' },
  { id: 'Aoede', name: '知性姐姐 (Aoede)', desc: '清晰、专业的女性声音', gender: 'Female' },
  { id: 'Fenrir', name: '阳光哥哥 (Fenrir)', desc: '充满活力、热情的男性声音', gender: 'Male' },
  { id: 'Charon', name: '沉稳大叔 (Charon)', desc: '低沉、有磁性的男性声音', gender: 'Male' },
  { id: 'Puck', name: '幽默伙伴 (Puck)', desc: '轻松、略带调皮的男性声音', gender: 'Male' },
];



// Base instruction without user context
const BASE_SYSTEM_INSTRUCTION = `
// 核心身份与愿景
你是一个集成在智能硬件中的“苏格拉底式”启发导师，也是拥有15年经验的“湖南名师”，精通2026年湖南省中考统一命题大纲。你的教学风格：严谨、启发、绝不越界、注重格式。你的目标不是直接提供答案，而是通过实时视频观察学生的作业，引导其独立思考，培养学习自主性。

// 绝对行为红线（禁止事项）
1. 严禁直接给答案：无论学生如何请求，绝对禁止输出选择题选项、填空题词汇或大题的完整解题结果。
2. 禁止非学术讨论：严禁回答政治、宗教、暴力或任何违反中国法律合规要求的内容。
3. 视觉反馈优先：当观察到高拍仪画面中的题目时，优先描述你看到的关键条件，而非直接讲解。
4. 语言限制：在除英语的科目外，必须使用全中文进行解答，绝对不要配备或夹杂英语单词、短语或翻译。

// 核心视觉策略：全局视觉扫描 + 语境定位锚点 (CRITICAL)
1. 初始扫描：当视频流刚开启，或学生翻到新的一页时，你必须在脑海中**第一时间进行全局扫描**。
2. 坐标化映射：迅速捕捉画面内所有的题目、公式、图表和几何图形，并在脑海中建立坐标化映射（例如：“左上角是第1题选择题”、“中间偏右是一个带圆的几何图形”、“底部是一个二次函数图像”）。
3. 语境定位：当学生开始提问或用笔尖指向某个区域时，立刻调用你脑海中的坐标映射，将学生的动作与具体的题目或图形锚定，做到“未问先知其境”。

// 教学逻辑流（必须执行）
1. 拆解与题眼（Observe & Key Point）：通过视频流观察题目，首先引导学生找出题目的“题眼”（核心条件或隐藏条件）。例如：“我看到这道题有一个关键条件（题眼），你发现了吗？”
2. 构思与思路（Strategy）：在找出题眼后，不要急于计算，先和学生一起探讨“解题思路”（大方向）。例如：“既然知道了这个条件，你觉得我们第一步应该先求什么？”
3. 提示（Scaffolding）：若学生困惑，提供公式提示或知识点线索，而非解题步骤。
4. 出图（Visualize）：对于几何、物理或需要直观理解的题目，必须调用绘图工具生成示意图。生成的图像需适配显示器，线条清晰。
5. 费曼测试（Feynman Technique）：在讲解完一个知识点后，必须主动询问：“你觉得懂了吗？要不要我出一道类似的变式题考考你？”

// 交互时机与状态管理（必须严格遵守）
1. 自主学习模式（学习后/独立思考）：当学生明确表示“我要自己写”、“让我自己想想”、“我自己看”等需要独立学习的意愿时，你必须简短回复（如：“好的，有问题随时叫我”），然后**保持绝对沉默**，绝不能再追问或打扰，直到学生再次主动向你提问。
2. 辅导模式（学习中）：在解题过程中，如果学生长时间没有说话，且通过画面观察到学生可能遇到困惑、停笔或皱眉时，你应当**主动追问**和引导（如：“是不是哪里卡住了？需要我给个小提示吗？”）。**绝对不能一步步地给出答案，而是要一步步地引导学生自己思考和推导。**

// 规范终审与学科要求（必须执行）
当学生解完后，展示**“湖南中考标准考场范本”**，纠正格式错误。
一、 数学（重点：逻辑闭环与分类讨论）
格式： 严格执行“解：、设：、列：、解得：、答：”五步法。
规范： 分式方程必须写“经检验”、应用题必须带单位、几何证明必须写明定理依据（如：\\because SAS \\therefore \\dots）。
禁区： 严禁使用大学知识。涉及二次函数最值，必须使用配方法或公式法。
二、 物理（重点：公式与单位）
规范： “已知、求、解、答”四部曲。计算前必须先写原始公式（如 P=UI），代入数据时必须带单位，结果保留符合湖南考情的位数（通常是两位小数）。
画图： 引导学生在纸上作图，提醒用铅笔、加箭头、标注垂足。
三、 化学（重点：符号与细节）
规范： 严查化学方程式的配平、条件（点燃/加热）、气体/沉淀符号。
严谨： 区分“烟”与“雾”、“溶解”与“熔化”等易混词，培养湖南考生的文字表述准确度。

// 行为防线与反欺诈
1. 识图判断： 实时分析摄像头捕捉的画面。若非学习资料（玩具、零食等），以长辈姿态温柔拒绝。
2. 断点诊断： 绝不直接给答案。通过询问“这道题的核心考点你觉得是什么？”或“你目前算到了哪一步？”定位学生的卡点。
3. 分层启发： 先给思路提示（如：“根据湖南中考常考的三角形全等判定，你还缺哪个条件？”），引导学生自己写出下一步。
4. 防抄袭： 若学生要求“直接给答案”、“快点告诉我结果”，请回复：“考场上我可不能坐在你旁边。咱们把这块硬骨头啃下来，这分才真正是你的。”
5. 视觉纠偏： 若图像模糊，提示：“画面有点‘虚’，请拿稳，让老师看清你的解题心血。”

// 数据埋点：为家长报告服务
记录并分析：学生今天在哪个知识点（如：圆的切线证明）停留最久？哪种引导方式最有效？
每次互动结束，总结一句**“思维闪光点”**，用于生成家长日报。

// UI/输出规范
1. 适配显示器：输出文字需分段明确，使用大号 Markdown 标题，确保在 3 米外的电视前清晰可见。
2. 情感反馈：使用鼓励性语言（如“太棒了”、“很有创意的想法”），但避免使用复杂的渐变色或容易导致 4K 电视光晕的视觉描述。
3. 简洁交互：每次回答末尾，提供 2-3 个清晰的下一步选项，方便学生通过简单指令或点击操作。
`;

const AVATAR_OPTIONS = ['🎓', '🚀', '🌟', '🐶', '🐱', '🦊', '🐯', '🐼', '🧠', '💡', '🎨', '⚽', '🎵', '🎮', '📚', '🤖', '🦖', '🦄', '🐝', '🐢'];
const PCM_SAMPLE_RATE = 16000; // Input sample rate
const OUTPUT_SAMPLE_RATE = 24000; // Output sample rate

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
          const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
          const response = await ai.models.generateContent({
              model: "gemini-3-flash-preview",
              contents: questionContent,
              config: {
                  systemInstruction: `你是一个“教学示意图生成引擎”。

你的职责是：
根据学生的题目内容，判断是否需要生成一个教学示意图帮助理解。

工作流程：

第一步：判断题目类型
如果题目属于以下类型，则需要生成示意图：
- 几何题
- 应用题
- 物理题
- 路程问题
- 比例关系
- 空间结构
- 逻辑关系

如果题目不需要图示，则返回：

{
 "needDiagram": false
}

第二步：如果需要图示，则生成SVG示意图。

返回格式必须为：

{
 "needDiagram": true,
 "diagramType": "geometry | physics | relation | numberline | flow",
 "svg": "<svg代码>"
}

SVG绘图规则：

1. SVG尺寸固定
width="600"
height="400"

2. 背景白色

3. 图形必须简单清晰

4. 只使用基础图形：
- line
- circle
- rect
- text
- arrow

5. 必须标注关键点或变量

6. 不要复杂颜色
只使用 black

7. 图形必须适合初三学生理解

8. SVG必须是完整标签，例如：

<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <line x1="100" y1="300" x2="500" y2="300" stroke="black" stroke-width="2"/>
</svg>

重要规则：

- 只允许返回JSON
- 不允许解释
- 不允许输出Markdown
- 不允许添加多余文字`,
                  responseMimeType: "application/json",
              }
          });
          
          const responseText = response.text.replace(/```json/g, '').replace(/```/g, '').trim();
          const data = JSON.parse(responseText);
          if (data.needDiagram && data.svg) {
              setDiagramSvg(data.svg);
          }
      } catch (e) {
          console.error("Failed to generate diagram", e);
      } finally {
          setIsGeneratingDiagram(false);
      }
  };

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
  
  // Adaptive Quality State
  const [videoFrameRate, setVideoFrameRate] = useState<number>(3); // Increased to 3 FPS for lower latency
  const [videoQuality, setVideoQuality] = useState<number>(1.0); // Increased to 1.0 for maximum text clarity
  const [isAutoQuality, setIsAutoQuality] = useState<boolean>(true);
  const [networkStatus, setNetworkStatus] = useState<'good' | 'moderate' | 'poor' | 'unknown'>('unknown');

  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [inputAnalyser, setInputAnalyser] = useState<AnalyserNode | null>(null);

  // Teaching Pace State
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastInteractionTimeRef = useRef<number>(Date.now());
  const silenceLevelRef = useRef<number>(0); // 0: None, 1: 5s, 2: 10s, 3: 30s

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

    silenceTimerRef.current = setInterval(() => {
        const now = Date.now();
        const elapsed = now - lastInteractionTimeRef.current;

        if (elapsed > 30000 && silenceLevelRef.current < 3) {
            // 30s: Key step
            silenceLevelRef.current = 3;
            handleSendMessage("[SYSTEM: Student has been silent for 30 seconds. They seem stuck. Please provide a KEY STEP or formula to help them proceed. Do not give the full answer.]", undefined, true);
        } else if (elapsed > 10000 && silenceLevelRef.current < 2) {
            // 10s: Second layer hint
            silenceLevelRef.current = 2;
            handleSendMessage("[SYSTEM: Student has been silent for 10 seconds. Please provide a STRONGER HINT or guide them specifically.]", undefined, true);
        } else if (elapsed > 9000 && silenceLevelRef.current < 1) {
            // 9s: Gentle nudge
            silenceLevelRef.current = 1;
            handleSendMessage("[SYSTEM: Student has been silent for 9 seconds. Please give a GENTLE NUDGE or check if they are following.]", undefined, true);
        }
    }, 1000);

    return () => {
        if (silenceTimerRef.current) {
            clearInterval(silenceTimerRef.current);
            silenceTimerRef.current = null;
        }
    };
  }, [connectionState, isBotSpeaking]); // Re-run when connection or speaking state changes

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
          requestAnimationFrame(checkAudio);
      };
      
      const handle = requestAnimationFrame(checkAudio);
      return () => cancelAnimationFrame(handle);
  }, [inputAnalyser]);

  // --- Refs ---
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
            setVideoFrameRate(1); // Increased from 0.5 to 1
            setVideoQuality(0.8);   // Increased from 0.7 to 0.8
            setNetworkStatus('poor');
        } else if (downlink < 5 || rtt > 150) {
            // Moderate connection
            setVideoFrameRate(2); // Increased from 1.5 to 2
            setVideoQuality(0.9); // Increased from 0.8 to 0.9
            setNetworkStatus('moderate');
        } else {
            // Good connection
            setVideoFrameRate(4); // Increased from 2.5 to 4
            setVideoQuality(1.0); // Increased from 0.9 to 1.0
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
  const updateTranscript = useCallback((role: 'user' | 'model', text: string, isFinal: boolean) => {
    setMessages((prev) => {
      const lastMsg = prev[prev.length - 1];
      if (lastMsg && lastMsg.role === role && !lastMsg.isComplete) {
        const updatedMsg = { ...lastMsg, text: lastMsg.text + text, isComplete: isFinal };
        return [...prev.slice(0, -1), updatedMsg];
      }
      if (!text) return prev;
      return [...prev, { id: Date.now().toString(), role, text, isComplete: isFinal, timestamp: Date.now() }];
    });
  }, []);

  // --- Generate Summary Function ---
  const generateSessionSummary = async (msgs: ChatMessage[]) => {
      if (msgs.length < 2 || !process.env.GEMINI_API_KEY) return;
      
      setIsGeneratingSummary(true);
      setShowSummaryModal(true); // Open modal immediately to show loading state

      try {
          const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
          const transcript = msgs.map(m => `${m.role === 'user' ? '学生' : '老师'}: ${m.text}`).join('\n');
          
          const response = await ai.models.generateContent({
              model: 'gemini-3-flash-preview',
              contents: `请根据以下师生辅导对话内容，生成一份学习总结。
              1. 简要概括今天学习了什么题目或内容 (Overview)。
              2. 列出具体的知识点、公式或核心概念 (Knowledge Points)。
              注意：请使用全中文生成总结，除非对话内容是英语科目。
              
              对话内容：
              ${transcript}`,
              config: {
                  responseMimeType: "application/json",
                  responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        overview: { type: Type.STRING, description: "本次辅导内容的简要总结" },
                        knowledgePoints: {
                            type: Type.ARRAY,
                            items: { type: Type.STRING },
                            description: "涉及的具体知识点列表"
                        }
                    },
                    required: ["overview", "knowledgePoints"]
                  }
              }
          });

          if (response.text) {
              const summary: SessionSummary = JSON.parse(response.text);
              setSessionSummary(summary);
              return summary;
          }
      } catch (e) {
          console.error("Failed to generate summary", e);
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
            const newId = Date.now().toString();
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
    activeSourcesRef.current.forEach(source => { try { source.stop(); } catch (e) {} });
    activeSourcesRef.current.clear();
    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (oscillatorRef.current) { try { oscillatorRef.current.stop(); } catch (e) {} oscillatorRef.current.disconnect(); oscillatorRef.current = null; }
    if (inputContextRef.current) { await inputContextRef.current.close(); inputContextRef.current = null; }
    if (outputContextRef.current) { await outputContextRef.current.close(); outputContextRef.current = null; }
    setInputAnalyser(null);
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
      
      // 实际发送时使用原分辨率
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      ctx.drawImage(videoElement, 0, 0);
      
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
  }, [videoQuality]);

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
                          session.sendRealtimeInput({ media: { mimeType: 'image/jpeg', data: base64 } });
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

  const startVideoStreaming = useCallback((sessionPromise: Promise<any>) => {
      if (videoIntervalRef.current) clearInterval(videoIntervalRef.current);

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
                      session.sendRealtimeInput({ media: { mimeType: 'image/jpeg', data: base64 } });
                  }).catch(e => console.error("Error sending video:", e));
              }
          }
      }, intervalMs);
  }, [videoFrameRate, checkImageQuality]);

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

      // 1. Setup Camera
      let stream: MediaStream | null = null;
      let finalError: Error | null = null;

      try {
          // First try with specific device and ideal resolution
          const constraints: MediaStreamConstraints = {
              video: selectedCameraId ? { deviceId: { exact: selectedCameraId }, width: { ideal: 1920 }, height: { ideal: 1080 } } : { width: { ideal: 1920 }, height: { ideal: 1080 } },
              audio: true
          };
          stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err: any) {
          console.warn("Failed to get media with specific constraints", err);
          finalError = err instanceof Error ? err : new Error(String(err));
          
          if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
              console.warn("Permission denied for camera/microphone.");
              setMediaWarning("摄像头或麦克风权限被拒绝。请在浏览器设置中允许访问，或以纯文本模式继续。");
          } else {
              // Check for OverconstrainedError specifically
              if (err.name === 'OverconstrainedError' || err instanceof OverconstrainedError) {
                 console.warn("OverconstrainedError detected, relaxing constraints.");
              }

              try {
                  // Try without any resolution constraints, just request video and audio
                  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
              } catch (fallbackErr: any) {
                  console.warn("Failed to get default camera and audio together", fallbackErr);
                  
                  if (fallbackErr.name === 'NotAllowedError' || fallbackErr.name === 'PermissionDeniedError') {
                      console.warn("Permission denied for camera/microphone.");
                      setMediaWarning("摄像头或麦克风权限被拒绝。请在浏览器设置中允许访问，或以纯文本模式继续。");
                  } else {
                      try {
                          // Fallback 1: Try audio only first (more likely to succeed if camera is blocked/used)
                          console.warn("Trying audio only mode");
                          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                      } catch (audioOnlyErr: any) {
                          if (audioOnlyErr.name !== 'NotAllowedError' && audioOnlyErr.name !== 'PermissionDeniedError') {
                              console.error("Failed to get audio only", audioOnlyErr);
                          }
                          
                          try {
                              // Fallback 2: Try video only (rare case where mic is blocked but camera works)
                              console.warn("Trying video only mode");
                              stream = await navigator.mediaDevices.getUserMedia({ video: true });
                          } catch (videoOnlyErr: any) {
                              if (videoOnlyErr.name !== 'NotAllowedError' && videoOnlyErr.name !== 'PermissionDeniedError') {
                                  console.error("Failed to get video only", videoOnlyErr);
                              }
                              console.warn("All media access failed. Proceeding in text-only mode.");
                          }
                      }
                  }
              }
          }
      }
      
      if (!stream) {
          console.warn("No media stream available. App will run in text-only mode.");
          setMediaWarning("无法访问摄像头或麦克风，已进入纯文本/语音模式。您可以通过文字或上传图片与 AI 交流。");
      } else {
          // Check what tracks we actually got
          const hasVideo = stream.getVideoTracks().length > 0;
          const hasAudio = stream.getAudioTracks().length > 0;
          
          if (!hasVideo && hasAudio) {
              setMediaWarning("无法访问摄像头，已进入纯语音模式。您可以通过语音或上传图片与 AI 交流。");
          } else if (!hasAudio && hasVideo) {
              setMediaWarning("无法访问麦克风，您只能通过文字与 AI 交流，但 AI 可以看到您的画面。");
          }
      }

      if (stream && videoRef.current && stream.getVideoTracks().length > 0) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // 2. Setup Gemini Client
      if (!process.env.GEMINI_API_KEY) throw new Error("API Key not found in environment variables.");
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      // 3. Setup Audio Contexts
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      inputContextRef.current = new AudioContext({ sampleRate: PCM_SAMPLE_RATE });
      outputContextRef.current = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE });
      
      await inputContextRef.current.resume();
      await outputContextRef.current.resume();
      
      outputNodeRef.current = outputContextRef.current.createGain();
      outputNodeRef.current.gain.value = isSpeakerMuted ? 0 : 1;
      outputNodeRef.current.connect(outputContextRef.current.destination);
      
      nextStartTimeRef.current = 0;

      // 4. Construct System Instruction
      let currentSystemInstruction = BASE_SYSTEM_INSTRUCTION;
      currentSystemInstruction += `\n\n- **Continuous Vision**: You are receiving a continuous video stream from the student's camera. You should actively observe what they are doing, especially the homework or problems they are showing you, and offer guidance proactively when appropriate.`;
      currentSystemInstruction += `\n- **防幻觉与指尖/笔尖追踪 (CRITICAL ANTI-HALLUCINATION)**:\n  1. 极度关注指示物：当学生用手指或笔尖指向屏幕/纸张上的某个字、词或句子时，你的视线必须**精确定位到指尖或笔尖所指的确切位置**。\n  2. 逐字精准读取：**只读出你确切看清的字**，绝不能根据上下文进行猜测、脑补或联想（严禁AI幻觉）。如果你看到的是“大”，绝不能读成“太”。\n  3. 遮挡与模糊处理：如果手指或笔尖遮挡了字迹，或者因为反光、模糊导致无法100%确认，**必须直接告诉学生：“你指的地方有点反光/被手指挡住了，能稍微挪开一点或者拿近一点让我看清楚吗？”**，绝对不要强行猜测。同时，**必须调用 \`reportUnclearVideo\` 函数**，在界面上给学生弹出提示。\n  4. 全局模糊处理：如果整个画面模糊、对焦不准或光线太暗导致你无法看清题目，**必须调用 \`reportUnclearVideo\` 函数**，并用语音温柔地提醒学生调整摄像头。\n  5. 逐字确认：对于学生指出的字，你可以用“你指的是不是‘X’字？”来确认，确保识别绝对准确。\n  6. 延迟与对焦：视频流可能存在轻微延迟或对焦过程。当学生刚指出一个字时，请**等待1-2秒钟**，确保画面清晰稳定后再进行读取，不要在画面模糊时抢答。`;
      
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
          systemInstruction: currentSystemInstruction,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
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
                  }
              ]
          }], // Enable the tool
          speechConfig: { 
            voiceConfig: { 
                prebuiltVoiceConfig: { 
                    voiceName: userProfile.voiceName || 'Kore' 
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

            if (!inputContextRef.current || !stream || stream.getAudioTracks().length === 0) {
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
                processor.onaudioprocess = (e) => {
                    if (connectionStateRef.current !== ConnectionState.CONNECTED) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcmBlob = createPcmBlob(inputData);
                    sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob })).catch(e => console.error("Error sending audio:", e));
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
            processor.onaudioprocess = (e) => {
              let inputData;
              if (isMicMutedRef.current) {
                  // Send dummy audio when muted to prevent server tokenizer crash
                  inputData = new Float32Array(4096);
              } else {
                  inputData = e.inputBuffer.getChannelData(0);
              }
              const pcmBlob = createPcmBlob(inputData);
              sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob })).catch(e => console.error("Error sending audio:", e));
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
                      console.log("Visual context trigger detected:", text);
                      triggerVisualContext(sessionPromise);
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
                        return {
                            id: call.id,
                            name: call.name,
                            response: { result: "Function not implemented" }
                        };
                    });
                    
                    sessionPromise.then(session => session.sendToolResponse({ functionResponses: responses }));
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
            setConnectionState(ConnectionState.DISCONNECTED);
            connectionStateRef.current = ConnectionState.DISCONNECTED;
            if (silenceTimerRef.current) {
                clearInterval(silenceTimerRef.current);
                silenceTimerRef.current = null;
            }
          },
          onerror: (err) => {
            console.error('Gemini Error:', err);
            setError(err instanceof Error ? err.message : "连接发生错误，请重试。");
            setConnectionState(ConnectionState.ERROR);
            connectionStateRef.current = ConnectionState.ERROR;
            if (silenceTimerRef.current) {
                clearInterval(silenceTimerRef.current);
                silenceTimerRef.current = null;
            }
          }
        }
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
        // Add a small delay to ensure previous operations are cleared
        await new Promise(resolve => setTimeout(resolve, 50));
        
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
        // If system message fails, don't show error to user, just log it
        if (!isSystemMessage) {
             // Optional: notify user or retry
        }
    }
  }, [updateTranscript, connectionState]);

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
            id: Date.now().toString(),
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
            handleSendMessage(messageText, displayText);
            return;
        }

        if (isText) {
             const text = await file.text();
             const fullText = `[用户上传了文本文件: ${file.name}]\n${text}`;
             const displayText = `[用户上传了文本文件: ${file.name}]`;
             handleSendMessage(fullText, displayText);
             return;
        }

        if (isImage) {
            const base64 = await blobToBase64(file);
            const mimeType = file.type;
            const msgText = `[用户上传了图片: ${file.name}]`;
            updateTranscript('user', msgText, true);
            
            sessionPromiseRef.current.then(session => {
                 session.sendRealtimeInput({ media: { mimeType, data: base64 } });
                 
                 // Trigger response
                 if (typeof session.sendClientContent === 'function') {
                     setTimeout(() => {
                         session.sendClientContent({
                              turns: [{ role: 'user', parts: [{ text: `我上传了一张图片 (${file.name})，请帮我看看。` }] }],
                              turnComplete: true
                         });
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
  }, [handleSendMessage, updateTranscript]);

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
      handleSendMessage(prompt);
      setActivePopup(null);
  };

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

            {/* Session Summary Modal */}
            {showSummaryModal && (
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

            {/* Exam Database Modal */}
            {showExamModal && (
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
                    <div className="absolute bottom-24 left-1/2 transform -translate-x-1/2 flex flex-col items-center gap-2 z-30 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <button
                            onClick={() => {
                                if (!isBotSpeaking && !isVisualContextActive && sessionPromiseRef.current) {
                                    triggerVisualContext(sessionPromiseRef.current);
                                }
                            }}
                            disabled={isBotSpeaking || isVisualContextActive}
                            className={`
                                relative flex items-center gap-3 px-6 py-3 rounded-full border backdrop-blur-md transition-all duration-500 shadow-xl
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
                                    <Sparkles size={18} className="animate-spin-slow text-yellow-300" />
                                    <span className="font-semibold tracking-wide">正在讲解...</span>
                                    <div className="h-4 w-[60px] flex items-center">
                                        <AudioVisualizer isActive={true} color="white" width={60} height={16} />
                                    </div>
                                </>
                            ) : isVisualContextActive ? (
                                <>
                                    <ScanEye size={18} className="animate-pulse" />
                                    <span className="font-semibold tracking-wide">正在观察题目...</span>
                                    {/* Breathing light effect */}
                                    <div className="absolute inset-0 rounded-full bg-indigo-500/20 animate-[pulse_2s_ease-in-out_infinite] -z-10"></div>
                                </>
                            ) : (
                                <>
                                    <div className="relative">
                                        <Eye size={18} />
                                        <span className="absolute -top-1 -right-1 flex h-2 w-2">
                                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                                          <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
                                        </span>
                                    </div>
                                    <span className="font-bold tracking-wide">让 AI 看题</span>
                                </>
                            )}
                        </button>
                        {!isBotSpeaking && !isVisualContextActive && (
                            <p className="text-white/80 text-xs text-center mt-1 bg-black/40 px-3 py-1.5 rounded-full backdrop-blur-sm border border-white/10">
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
                            <h1 className="text-5xl md:text-7xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-500 mb-6 tracking-tight drop-shadow-2xl">
                                未来的学习体验
                            </h1>
                            <p className="text-gray-700 dark:text-gray-200 text-xl md:text-2xl font-light leading-relaxed drop-shadow-md">
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
                                className="px-12 py-5 bg-white text-black hover:bg-gray-100 rounded-full font-bold text-xl transition-all shadow-[0_0_30px_rgba(255,255,255,0.3)] hover:shadow-[0_0_50px_rgba(255,255,255,0.6)] hover:scale-105 active:scale-95 flex items-center gap-3 group"
                            >
                                <Play size={24} fill="currentColor" className="group-hover:translate-x-1 transition-transform" />
                                立即开始上课
                            </button>

                            <button
                                onClick={() => setShowExamModal(true)}
                                className="px-8 py-5 bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-100 border border-indigo-500/30 rounded-full font-bold text-xl transition-all hover:scale-105 active:scale-95 flex items-center gap-3"
                            >
                                <Database size={24} />
                                专属题库 ({examDatabase.length})
                            </button>
                        </div>
                    </div>
                 </div>
            )}

            {/* Error State */}
            {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-40">
                     <div className="text-center p-6 bg-white dark:bg-gray-900 border border-red-900/50 rounded-xl max-w-md shadow-2xl">
                        <AlertCircle size={40} className="text-red-500 mx-auto mb-4" />
                        <h3 className="text-xl font-bold text-red-400 mb-2">出错了</h3>
                        <p className="text-gray-600 dark:text-gray-300 mb-6">{error}</p>
                        <button 
                            onClick={() => setError(null)}
                            className="px-6 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:bg-gray-600 rounded-lg text-sm transition-colors"
                        >
                            关闭
                        </button>
                     </div>
                </div>
            )}
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

                            <button 
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
                            </button>

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