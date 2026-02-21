import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type } from '@google/genai';
import { Video, Mic, MicOff, Play, Square, AlertCircle, Volume2, Sparkles, Eye, Settings, VolumeX, RefreshCw, Camera, FlipHorizontal, Lightbulb, Key, X, MessageCircleQuestion, ArrowRight, ScanEye, Target, UserRoundPen, Check, ChevronRight, Gauge, Save, AudioLines, Wifi, WifiOff, FileText, Loader2, BookOpen } from 'lucide-react';
import { ConnectionState, ChatMessage, SavedSession, UserProfile, SessionSummary } from '../types';
import { createPcmBlob, decode, decodeAudioData, blobToBase64 } from '../utils/audioUtils';
import AudioVisualizer from './AudioVisualizer';
import Transcript from './Transcript';
import DiagramBoard, { DiagramData } from './DiagramBoard';
import * as pdfjsLib from 'pdfjs-dist';

// Set worker source for PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

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

// Tool Declaration for Diagramming
const drawDiagramTool = {
  functionDeclarations: [
    {
      name: "draw_diagram",
      description: "Draw a simple 2D diagram (geometry, physics, chemistry) to visualize a concept. Use this when a visual aid would help the student understand.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          title: { type: Type.STRING, description: "Title of the diagram" },
          description: { type: Type.STRING, description: "Short description of what this diagram shows" },
          viewBox: { type: Type.STRING, description: "SVG viewBox, e.g., '0 0 400 300'" },
          shapes: {
            type: Type.ARRAY,
            description: "List of shapes to draw",
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING, description: "Unique ID for the shape" },
                type: { type: Type.STRING, description: "Type of shape: 'line', 'circle', 'rect', 'text', 'polygon', 'arrow'" },
                x: { type: Type.NUMBER },
                y: { type: Type.NUMBER },
                x1: { type: Type.NUMBER },
                y1: { type: Type.NUMBER },
                x2: { type: Type.NUMBER },
                y2: { type: Type.NUMBER },
                r: { type: Type.NUMBER },
                width: { type: Type.NUMBER },
                height: { type: Type.NUMBER },
                points: { type: Type.STRING, description: "Points for polygon, e.g., '0,0 100,0 50,100'" },
                content: { type: Type.STRING, description: "Text content" },
                color: { type: Type.STRING, description: "Stroke color (hex or name)" },
                fill: { type: Type.STRING, description: "Fill color (hex or name)" },
                label: { type: Type.STRING, description: "Label text for the shape" }
              },
              required: ["id", "type"]
            }
          }
        },
        required: ["title", "shapes"]
      }
    }
  ]
};

// Base instruction without user context
const BASE_SYSTEM_INSTRUCTION = `
You are a friendly, encouraging, and expert tutor. 
Your goal is to help the user solve problems they show you on video.

CRITICAL RULE: NEVER GIVE THE FINAL ANSWER DIRECTLY. Instead, guide the student step-by-step to find the answer themselves using the Socratic method.

Interaction Protocol:
1. **Visual Attention (Pointing Recognition)**: 
   - ACTIVELY look for a human finger, a pen, or any pointing object in the video frame.
   - **IMMEDIATE VERBAL ACKNOWLEDGMENT**: If you see a gesture/pointing, start your response by confirming it (e.g., "我看到你指着...", "I see you are pointing at...").
   - Focus your analysis EXCLUSIVELY on the specific problem, equation, or text diagram being pointed at by the tip.
   - If multiple problems are visible, assume the user wants help with the one they are pointing to.

2. **Identify Category**: When you see the problem, verbally identify its type/subject (e.g., "这是一道[数学/物理/编程]题...").

3. **Problem Analysis (Knowledge & Key Point)**: Before solving, you MUST explicitly state:
   - **Knowledge Points (知识点)**: What specific academic concepts or formulas are being tested? (e.g., "这道题主要考察的知识点是勾股定理和相似三角形...").
   - **The "Eye" of the Problem (题眼)**: What is the critical trick, hidden condition, or key step that breaks the problem open? (e.g., "这道题的题眼在于要注意题目中提到的'匀速直线运动'，这意味着合力为零").

4. **Step-by-Step Guidance**: 
   - Ask the student what they think the first step is based on the "Eye" of the problem.
   - If they are stuck, provide a small hint linking back to the Knowledge Points.
   - Verify their understanding before moving to the next step.
   - If they calculate incorrectly, gently ask them to double-check.

5. **Visual Checks**: If the text is blurry, too small, or cut off, explicitly ask the user to adjust the camera, hold the paper steady, or move closer. Use phrases like "看不清", "模糊", or "调整摄像头".
6. **Language**: Speak in Chinese (Mandarin) unless the user speaks to you in another language.
7. **Tone**: Be supportive, patient, and concise. Celebrate small wins when the user gets a step right.

8. **Closing & Confirmation (MANDATORY)**:
   - **Scenario A (Explanation Finished)**: When you finish explaining a concept, DO NOT stop silently. You MUST ask: "这道题你现在真的弄懂了吗？" (Do you really understand this problem now?).
   - **Scenario B (User Gets Answer Right)**: When the user calculates or states the correct answer, acknowledge it (e.g., "对了!"), but IMMEDIATELY follow up with a deep-dive check: "答案是对的，但这道题的原理你真的完全弄懂了吗？" or "你能再跟我讲一遍为什么选这个吗？我想确认你真的懂了。"
   - **Goal**: Ensure deep understanding, not just correct answers.

9. **Text Input Handling**: If the user sends a text message or question (e.g. via the chat input), YOU MUST RESPOND TO IT IMMEDIATELY. Do not ignore text input. You can combine text questions with visual context if relevant.

10. **Strict Scope Enforcement**: You are strictly a learning assistant. Do NOT discuss topics unrelated to education, learning, or problem-solving. If a user asks about non-learning topics (e.g., weather, jokes, news, personal life), politely redirect them: "我只负责学习辅导哦，让我们回到题目上来吧。"

11. **Visual Aids**: Use the 'draw_diagram' tool whenever a visual explanation would be helpful (e.g., geometry figures, force diagrams, circuit diagrams). Explain what you are drawing while you draw it.
`;

const AVATAR_OPTIONS = ['🎓', '🚀', '🌟', '🐶', '🐱', '🦊', '🐯', '🐼', '🧠', '💡', '🎨', '⚽', '🎵', '🎮', '📚', '🤖', '🦖', '🦄', '🐝', '🐢'];
const PCM_SAMPLE_RATE = 16000; // Input sample rate
const OUTPUT_SAMPLE_RATE = 24000; // Output sample rate

const LiveTutor: React.FC = () => {
  // --- State ---
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Profile State
  const [userProfile, setUserProfile] = useState<UserProfile>({ 
      name: '', 
      age: '', 
      avatar: '🎓',
      voiceName: 'Kore' // Default voice
  });
  const [showProfileModal, setShowProfileModal] = useState(false);

  // Insight & Summary State
  const [insightData, setInsightData] = useState<{ knowledge: string | null; eye: string | null }>({ knowledge: null, eye: null });
  const [activePopup, setActivePopup] = useState<{ type: 'knowledge' | 'eye'; content: string } | null>(null);
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [diagramData, setDiagramData] = useState<DiagramData | null>(null);
  
  const [showBlurWarning, setShowBlurWarning] = useState(false);
  const [scannerActive, setScannerActive] = useState(false);
  const [showSaveConfirm, setShowSaveConfirm] = useState(false);

  // Media & Network State
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [isSpeakerMuted, setIsSpeakerMuted] = useState(false);
  const [isVideoMirrored, setIsVideoMirrored] = useState(true); 
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [showSettings, setShowSettings] = useState(false);
  
  // Adaptive Quality State
  const [videoFrameRate, setVideoFrameRate] = useState<number>(2); // Default 2 FPS
  const [videoQuality, setVideoQuality] = useState<number>(0.6); // Default JPEG quality 0.6
  const [isAutoQuality, setIsAutoQuality] = useState<boolean>(true);
  const [networkStatus, setNetworkStatus] = useState<'good' | 'moderate' | 'poor' | 'unknown'>('unknown');

  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [inputAnalyser, setInputAnalyser] = useState<AnalyserNode | null>(null);

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
  const nextStartTimeRef = useRef<number>(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const frameIntervalRef = useRef<number | null>(null);

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
            setVideoFrameRate(0.5); // 1 frame every 2 seconds
            setVideoQuality(0.4);   // Higher compression
            setNetworkStatus('poor');
        } else if (downlink < 5 || rtt > 150) {
            // Moderate connection
            setVideoFrameRate(1.5);
            setVideoQuality(0.5);
            setNetworkStatus('moderate');
        } else {
            // Good connection
            setVideoFrameRate(2.5);
            setVideoQuality(0.65);
            setNetworkStatus('good');
        }
    };

    connection.addEventListener('change', updateQuality);
    updateQuality(); // Initial check

    return () => connection.removeEventListener('change', updateQuality);
  }, [isAutoQuality]);

  // Load Profile on Mount
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
      } catch (e) {
          console.error("Failed to load profile", e);
      }
  }, []);

  // Save Profile Helper
  const saveProfile = (newProfile: UserProfile) => {
      setUserProfile(newProfile);
      localStorage.setItem('user_profile', JSON.stringify(newProfile));
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
        await navigator.mediaDevices.getUserMedia({ audio: true, video: true }); // Request perm first
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        setVideoDevices(cameras);
        if (cameras.length > 0) {
            const backCamera = cameras.find(c => c.label.toLowerCase().includes('back') || c.label.toLowerCase().includes('environment'));
            setSelectedCameraId(backCamera ? backCamera.deviceId : cameras[0].deviceId);
            if (backCamera) setIsVideoMirrored(false);
        }
      } catch (e) {
        console.error("Error enumerating devices", e);
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
            const newStream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: deviceId }, width: 1280, height: 720 },
                audio: false 
            });
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
      if (msgs.length < 2 || !process.env.API_KEY) return;
      
      setIsGeneratingSummary(true);
      setShowSummaryModal(true); // Open modal immediately to show loading state

      try {
          const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
          const transcript = msgs.map(m => `${m.role === 'user' ? '学生' : '老师'}: ${m.text}`).join('\n');
          
          const response = await ai.models.generateContent({
              model: 'gemini-2.5-flash',
              contents: `请根据以下师生辅导对话内容，生成一份学习总结。
              1. 简要概括今天学习了什么题目或内容 (Overview)。
              2. 列出具体的知识点、公式或核心概念 (Knowledge Points)。
              
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
    activeSourcesRef.current.forEach(source => { try { source.stop(); } catch (e) {} });
    activeSourcesRef.current.clear();
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

  // --- Helper: Video Streaming ---
  const startVideoStreaming = useCallback((sessionPromise: Promise<any>) => {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    frameIntervalRef.current = window.setInterval(() => {
        if (!videoRef.current || !canvasRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx || video.readyState < 2) return;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        
        // Dynamic Quality Used Here
        canvas.toBlob(async (blob) => {
            if (blob) {
                const base64 = await blobToBase64(blob);
                sessionPromise.then(session => {
                    session.sendRealtimeInput({ media: { mimeType: 'image/jpeg', data: base64 } });
                });
            }
        }, 'image/jpeg', videoQuality); // Use state for quality
    }, 1000 / videoFrameRate); // Use state for FPS
  }, [videoFrameRate, videoQuality]);

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
      setScannerActive(false);
      currentSessionIdRef.current = null;

      // 1. Setup Camera
      const constraints: MediaStreamConstraints = {
          video: selectedCameraId ? { deviceId: { exact: selectedCameraId }, width: 1280, height: 720 } : { width: 1280, height: 720 },
          audio: true
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // 2. Setup Gemini Client
      if (!process.env.API_KEY) throw new Error("API Key not found in environment variables.");
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // 3. Setup Audio Contexts
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      inputContextRef.current = new AudioContext({ sampleRate: PCM_SAMPLE_RATE });
      outputContextRef.current = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE });
      
      outputNodeRef.current = outputContextRef.current.createGain();
      outputNodeRef.current.gain.value = isSpeakerMuted ? 0 : 1;
      outputNodeRef.current.connect(outputContextRef.current.destination);
      
      nextStartTimeRef.current = 0;

      // 4. Construct System Instruction
      let currentSystemInstruction = BASE_SYSTEM_INSTRUCTION;
      if (userProfile.name) {
          currentSystemInstruction += `\n11. **Student Profile**: The student's name is "${userProfile.name}". Use their name occasionally to be friendly.`;
      }
      if (userProfile.age) {
          currentSystemInstruction += `\n12. **Age Appropriateness**: The student is ${userProfile.age} years old. ADJUST YOUR EXPLANATION COMPLEXITY AND TONE TO MATCH A ${userProfile.age}-YEAR-OLD CHILD.`;
      } else {
          currentSystemInstruction += `\n12. **Age Appropriateness**: Assume the student is a middle school student. Explain concepts clearly and simply.`;
      }

      // Inject Past History Context
      try {
          const historyData = localStorage.getItem('tutoring_history');
          if (historyData) {
              const history: SavedSession[] = JSON.parse(historyData);
              // Take the last 5 sessions to keep context manageable
              const recentHistory = history.slice(0, 5).map(s => {
                  const date = new Date(s.timestamp).toLocaleDateString();
                  const summary = s.summary 
                      ? `Topic: ${s.summary.overview}, Key Points: ${s.summary.knowledgePoints.join(', ')}`
                      : `Content Preview: ${s.preview}`;
                  return `- [${date}]: ${summary}`;
              }).join('\n');

              if (recentHistory) {
                  currentSystemInstruction += `\n\n13. **Past Learning Context**: Here is a summary of the student's recent learning history. USE THIS to make connections (e.g., "Remember when we learned about [Topic] last time? This is similar...").\n${recentHistory}`;
              }
          }
      } catch (e) {
          console.error("Failed to load history for context", e);
      }

      // 5. Connect to Gemini Live
      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: currentSystemInstruction,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          tools: [drawDiagramTool], // Enable the tool
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
            
            if (!inputContextRef.current) return;
            const source = inputContextRef.current.createMediaStreamSource(stream);
            
            const analyser = inputContextRef.current.createAnalyser();
            analyser.fftSize = 64;
            analyser.smoothingTimeConstant = 0.5;
            source.connect(analyser);
            setInputAnalyser(analyser);

            const processor = inputContextRef.current.createScriptProcessor(4096, 1, 1);
            processor.onaudioprocess = (e) => {
              if (isMicMuted) return;
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createPcmBlob(inputData);
              sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob }));
            };
            source.connect(processor);
            processor.connect(inputContextRef.current.destination);

            startVideoStreaming(sessionPromise);
          },
          onmessage: async (msg: LiveServerMessage) => {
            if (msg.serverContent?.inputTranscription) {
              const text = msg.serverContent.inputTranscription.text;
              if (text) updateTranscript('user', text, false);
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
                        if (call.name === 'draw_diagram') {
                            try {
                                const args = call.args as unknown as DiagramData;
                                console.log("Diagram Tool Called:", args);
                                setDiagramData(args);
                                return {
                                    id: call.id,
                                    name: call.name,
                                    response: { result: "Diagram displayed successfully." }
                                };
                            } catch (e) {
                                console.error("Error processing diagram tool:", e);
                                return {
                                    id: call.id,
                                    name: call.name,
                                    response: { error: "Failed to render diagram." }
                                };
                            }
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
          },
          onerror: (err) => {
            console.error('Gemini Error:', err);
            setError(err instanceof Error ? err.message : "连接发生错误，请重试。");
            setConnectionState(ConnectionState.ERROR);
          }
        }
      });
      sessionPromiseRef.current = sessionPromise;

    } catch (err: any) {
      console.error(err);
      setError(err.message || "无法启动会话");
      setConnectionState(ConnectionState.ERROR);
      stopSession();
    }
  };

  const handleSendMessage = useCallback((text: string, displayOverride?: string) => {
    if (!sessionPromiseRef.current) return;
    updateTranscript('user', displayOverride || text, true);
    sessionPromiseRef.current.then(session => {
        const s = session as any;
        if (typeof s.send === 'function') {
            s.send({
                 clientContent: {
                     turns: [{ role: 'user', parts: [{ text }] }],
                     turnComplete: true
                 }
            });
        }
    }).catch(err => console.error("Failed to send text message:", err));
  }, [updateTranscript]);

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
                 const s = session as any;
                 if (typeof s.send === 'function') {
                     setTimeout(() => {
                         s.send({
                              clientContent: {
                                  turns: [{ role: 'user', parts: [{ text: `我上传了一张图片 (${file.name})，请帮我看看。` }] }],
                                  turnComplete: true
                              }
                         });
                     }, 200);
                 }
            });
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
    <div className="flex h-screen w-full bg-gray-950 text-white overflow-hidden">
      {/* CSS Animations */}
      <style>{`
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
                        <Video size={20} className="text-white" />
                    </div>
                    <h1 className="text-xl font-bold tracking-tight">Live Tutor</h1>
                </div>
                
                <div className="flex items-center gap-4">
                     <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-black/40 backdrop-blur-md border border-white/10">
                        <span className={`w-2 h-2 rounded-full ${
                            connectionState === ConnectionState.CONNECTED ? 'bg-green-500 animate-pulse' : 
                            connectionState === ConnectionState.CONNECTING ? 'bg-yellow-500' : 'bg-red-500'
                        }`}></span>
                        <span className="text-xs font-medium text-gray-300">
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
                            <h3 className="font-bold text-xl mb-1 text-white">画面有点模糊</h3>
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
                            className="bg-gray-900/80 backdrop-blur-lg border border-indigo-500/40 p-4 rounded-2xl shadow-2xl border-l-4 border-l-indigo-500 transform transition-all hover:scale-105 cursor-pointer hover:bg-gray-800/90 group"
                        >
                            <div className="flex items-center gap-2 mb-2 text-indigo-300 font-bold text-sm tracking-wide group-hover:text-indigo-200">
                                <Lightbulb size={18} className="fill-indigo-500/20" />
                                <span>核心知识点</span>
                            </div>
                            <p className="text-gray-100 text-sm leading-relaxed font-medium line-clamp-2">{insightData.knowledge}</p>
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
                            className="bg-gray-900/80 backdrop-blur-lg border border-emerald-500/40 p-4 rounded-2xl shadow-2xl border-l-4 border-l-emerald-500 transform transition-all hover:scale-105 cursor-pointer hover:bg-gray-800/90 group" 
                            style={{animationDelay: '150ms'}}
                        >
                            <div className="flex items-center gap-2 mb-2 text-emerald-300 font-bold text-sm tracking-wide group-hover:text-emerald-200">
                                <Key size={18} className="fill-emerald-500/20" />
                                <span>解题关键 (题眼)</span>
                            </div>
                            <p className="text-gray-100 text-sm leading-relaxed font-medium line-clamp-2">{insightData.eye}</p>
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
                    <div className="bg-gray-900 border border-gray-700 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
                        {/* Popup Header */}
                        <div className={`p-4 flex justify-between items-center ${activePopup.type === 'knowledge' ? 'bg-indigo-900/30' : 'bg-emerald-900/30'} border-b border-white/5`}>
                             <div className={`flex items-center gap-2 font-bold ${activePopup.type === 'knowledge' ? 'text-indigo-300' : 'text-emerald-300'}`}>
                                {activePopup.type === 'knowledge' ? <Lightbulb size={20} /> : <Key size={20} />}
                                <span>{activePopup.type === 'knowledge' ? '核心知识点' : '解题关键'}</span>
                             </div>
                             <button onClick={() => setActivePopup(null)} className="p-1 hover:bg-white/10 rounded-full text-gray-400 hover:text-white transition-colors">
                                <X size={20} />
                             </button>
                        </div>
                        
                        {/* Popup Content */}
                        <div className="p-6">
                            <p className="text-lg text-gray-100 leading-relaxed font-medium">{activePopup.content}</p>
                        </div>

                        {/* Popup Footer (Actions) */}
                        <div className="p-4 bg-black/20 flex gap-3">
                             <button 
                                onClick={handleAskExplain}
                                className="flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold transition-all shadow-lg shadow-indigo-600/20 active:scale-95"
                             >
                                <MessageCircleQuestion size={18} />
                                让 AI 详细讲解
                             </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Session Summary Modal */}
            {showSummaryModal && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-in fade-in duration-300">
                    <div className="bg-gray-900 border border-indigo-500/30 w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300 flex flex-col max-h-[80vh]">
                        {/* Header */}
                        <div className="p-5 border-b border-gray-700/50 bg-gradient-to-r from-indigo-900/40 to-gray-900 flex justify-between items-center">
                            <div className="flex items-center gap-3">
                                <div className="bg-indigo-600 p-2 rounded-lg shadow-lg shadow-indigo-600/20">
                                    <BookOpen size={24} className="text-white" />
                                </div>
                                <div>
                                    <h3 className="font-bold text-xl text-white">今日学习报告</h3>
                                    <p className="text-xs text-indigo-300">AI 智能生成的辅导总结</p>
                                </div>
                            </div>
                            <button 
                                onClick={() => setShowSummaryModal(false)} 
                                className="p-2 hover:bg-white/10 rounded-full text-gray-400 hover:text-white transition-colors"
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
                                    <p className="text-gray-400 animate-pulse">正在整理学习笔记，请稍候...</p>
                                </div>
                            ) : sessionSummary ? (
                                <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                                    {/* Overview Section */}
                                    <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                                        <h4 className="text-indigo-400 text-sm font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
                                            <FileText size={16} /> 学习概览
                                        </h4>
                                        <p className="text-gray-200 leading-relaxed text-sm">
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
                                                <li key={idx} className="flex gap-3 items-start bg-gray-800/30 p-3 rounded-lg border border-gray-700/50 hover:border-emerald-500/30 transition-colors">
                                                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xs font-bold mt-0.5">
                                                        {idx + 1}
                                                    </span>
                                                    <span className="text-gray-300 text-sm">{point}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                    
                                    <div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg text-xs text-blue-300 text-center">
                                        这份报告已保存到历史记录中，随时可以在左侧回顾。
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center text-gray-500 py-10">
                                    <p>暂无总结内容</p>
                                </div>
                            )}
                        </div>

                        {/* Footer */}
                        <div className="p-4 bg-gray-800/50 border-t border-gray-700/50 flex justify-end">
                            <button 
                                onClick={() => setShowSummaryModal(false)}
                                disabled={isGeneratingSummary}
                                className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
                        className={`p-3 rounded-full text-white backdrop-blur-md border border-white/10 transition-all ${isVideoMirrored ? 'bg-indigo-600/80 hover:bg-indigo-600' : 'bg-black/50 hover:bg-black/70'}`}
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
                        className="p-3 rounded-full bg-black/50 hover:bg-black/70 text-white backdrop-blur-md border border-white/10 transition-all group relative"
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
                        className="p-3 rounded-full bg-black/50 hover:bg-black/70 text-white backdrop-blur-md border border-white/10 transition-all"
                        title="设置"
                    >
                        <Settings size={20} />
                    </button>
                    
                    {/* Settings Dropdown */}
                    {showSettings && (
                        <div className="flex flex-col gap-2 p-3 rounded-2xl bg-black/60 backdrop-blur-xl border border-white/10 animate-in fade-in zoom-in duration-200 w-64 shadow-2xl">
                            <div className="text-xs font-semibold text-gray-400 mb-1 uppercase tracking-wider">摄像头设置</div>
                            
                            {/* Device Selector */}
                            <div className="relative mb-2">
                                <select 
                                    value={selectedCameraId}
                                    onChange={(e) => switchCamera(e.target.value)}
                                    className="w-full bg-gray-800/80 text-white text-sm rounded-lg p-2 pl-8 outline-none border border-gray-700 hover:border-indigo-500 appearance-none"
                                >
                                    {videoDevices.map(device => (
                                        <option key={device.deviceId} value={device.deviceId}>
                                            {device.label || `摄像头 ${device.deviceId.slice(0, 5)}...`}
                                        </option>
                                    ))}
                                </select>
                                <Camera size={14} className="absolute left-2.5 top-2.5 text-gray-400 pointer-events-none" />
                            </div>

                            {/* Smart Quality Toggle */}
                            <div className="mb-2">
                                 <button 
                                    onClick={() => setIsAutoQuality(!isAutoQuality)}
                                    className={`w-full flex items-center justify-between p-2 rounded-lg text-sm transition-colors ${isAutoQuality ? 'bg-emerald-600/30 text-emerald-200' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
                                 >
                                     <div className="flex items-center gap-2">
                                         <Wifi size={16} />
                                         <span>智能画质调节</span>
                                     </div>
                                     <div className={`w-8 h-4 rounded-full relative transition-colors ${isAutoQuality ? 'bg-emerald-500' : 'bg-gray-600'}`}>
                                         <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${isAutoQuality ? 'left-4.5' : 'left-0.5'}`} style={{left: isAutoQuality ? '18px' : '2px'}} />
                                     </div>
                                 </button>
                                 {isAutoQuality && (
                                    <div className="mt-1 px-2 flex justify-between items-center text-[10px]">
                                        <span className="text-gray-500">当前网络:</span>
                                        <span className={`font-medium ${
                                            networkStatus === 'good' ? 'text-green-400' : 
                                            networkStatus === 'moderate' ? 'text-yellow-400' : 
                                            networkStatus === 'poor' ? 'text-red-400' : 'text-gray-400'
                                        }`}>
                                            {networkStatus === 'good' ? '极佳' : networkStatus === 'moderate' ? '一般' : networkStatus === 'poor' ? '较差' : '未知'}
                                        </span>
                                    </div>
                                 )}
                            </div>

                            {/* Frame Rate / Speed Slider */}
                            <div className={`mb-3 px-1 transition-opacity ${isAutoQuality ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
                                <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
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
                                    className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all"
                                />
                                <div className="flex justify-between text-[10px] text-gray-600 mt-1 font-medium">
                                    <span>省流</span>
                                    <span>极速</span>
                                </div>
                            </div>

                            <div className="h-px bg-gray-700/50 my-2" />

                            {/* Mirror Toggle */}
                            <button 
                                onClick={() => setIsVideoMirrored(!isVideoMirrored)}
                                className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors ${isVideoMirrored ? 'bg-indigo-600/30 text-indigo-200' : 'hover:bg-gray-700/50'}`}
                            >
                                <div className="flex items-center gap-2">
                                    <RefreshCw size={16} />
                                    <span>镜像画面</span>
                                </div>
                                <div className={`w-8 h-4 rounded-full relative transition-colors ${isVideoMirrored ? 'bg-indigo-500' : 'bg-gray-600'}`}>
                                    <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${isVideoMirrored ? 'left-4.5' : 'left-0.5'}`} style={{left: isVideoMirrored ? '18px' : '2px'}} />
                                </div>
                            </button>
                            
                            {/* Speaker Mute Toggle */}
                            <button 
                                onClick={() => setIsSpeakerMuted(!isSpeakerMuted)}
                                className={`flex items-center justify-between p-2 rounded-lg text-sm transition-colors ${isSpeakerMuted ? 'bg-red-500/20 text-red-200' : 'hover:bg-gray-700/50'}`}
                            >
                                <div className="flex items-center gap-2">
                                    {isSpeakerMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                                    <span>AI 语音播放</span>
                                </div>
                                <div className={`w-8 h-4 rounded-full relative transition-colors ${!isSpeakerMuted ? 'bg-green-500' : 'bg-gray-600'}`}>
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
                    <div className="bg-gray-900 border border-gray-700 w-full max-w-sm rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200 max-h-[90vh] flex flex-col">
                        <div className="p-4 border-b border-gray-700/50 flex justify-between items-center bg-gray-800/50 flex-shrink-0">
                            <h3 className="font-bold text-lg text-white flex items-center gap-2">
                                <UserRoundPen size={20} className="text-indigo-400" />
                                个人资料设置
                            </h3>
                            <button onClick={() => setShowProfileModal(false)} className="p-1 hover:bg-white/10 rounded-full text-gray-400 hover:text-white transition-colors">
                                <X size={20} />
                            </button>
                        </div>
                        
                        <div className="p-6 space-y-6 overflow-y-auto">
                            {/* Avatar Selection */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-2">选择头像</label>
                                <div className="grid grid-cols-5 gap-2">
                                    {AVATAR_OPTIONS.map(avatar => (
                                        <button
                                            key={avatar}
                                            onClick={() => saveProfile({...userProfile, avatar})}
                                            className={`h-10 w-10 flex items-center justify-center text-xl rounded-full transition-all ${
                                                userProfile.avatar === avatar 
                                                ? 'bg-indigo-600 ring-2 ring-indigo-300 ring-offset-2 ring-offset-gray-900' 
                                                : 'bg-gray-800 hover:bg-gray-700'
                                            }`}
                                        >
                                            {avatar}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Name Input */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-1">昵称 / 名字</label>
                                <input
                                    type="text"
                                    value={userProfile.name}
                                    onChange={(e) => saveProfile({...userProfile, name: e.target.value})}
                                    placeholder="比如: 小明"
                                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                                />
                            </div>

                            {/* Age Input */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-1">年龄 (岁)</label>
                                <input
                                    type="number"
                                    value={userProfile.age}
                                    onChange={(e) => saveProfile({...userProfile, age: e.target.value})}
                                    placeholder="AI 将根据年龄调整讲解难度"
                                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                                />
                                <p className="text-xs text-gray-500 mt-1">设置年龄后，AI 会使用更适合该年龄段的语言。</p>
                            </div>

                            {/* Voice Selection */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-2">选择 AI 语音</label>
                                <div className="grid grid-cols-1 gap-2 max-h-40 overflow-y-auto pr-1">
                                    {VOICE_OPTIONS.map(voice => (
                                        <button
                                            key={voice.id}
                                            onClick={() => saveProfile({...userProfile, voiceName: voice.id})}
                                            className={`flex items-center justify-between px-4 py-3 rounded-xl border transition-all ${
                                                (userProfile.voiceName || 'Kore') === voice.id 
                                                ? 'bg-indigo-600/20 border-indigo-500 text-white' 
                                                : 'bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700 hover:border-gray-600'
                                            }`}
                                        >
                                            <div className="flex flex-col items-start">
                                                <div className="flex items-center gap-2">
                                                    <span className={`text-sm font-semibold ${(userProfile.voiceName || 'Kore') === voice.id ? 'text-indigo-300' : 'text-gray-300'}`}>
                                                        {voice.name}
                                                    </span>
                                                    {voice.gender === 'Female' ? <span className="text-xs text-rose-400 bg-rose-900/30 px-1 rounded">女声</span> : <span className="text-xs text-blue-400 bg-blue-900/30 px-1 rounded">男声</span>}
                                                </div>
                                                <span className="text-xs text-gray-500 mt-1">{voice.desc}</span>
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

                        <div className="p-4 bg-gray-800/30 border-t border-gray-700/50 flex-shrink-0">
                            <button 
                                onClick={() => setShowProfileModal(false)}
                                className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                            >
                                <Check size={16} />
                                保存并关闭
                            </button>
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

                    {/* Diagram Board */}
                    <DiagramBoard 
                        data={diagramData} 
                        onClose={() => setDiagramData(null)} 
                    />

                    {/* Status Indicator (Bot) */}
                    <div className="absolute bottom-32 left-1/2 transform -translate-x-1/2 flex flex-col items-center gap-2 z-20 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className={`
                            relative flex items-center gap-3 px-6 py-3 rounded-full border backdrop-blur-md transition-all duration-500 shadow-xl
                            ${isBotSpeaking 
                                ? 'bg-emerald-500 border-emerald-400 text-white ripple-effect scale-110' 
                                : 'bg-gray-900/80 border-indigo-500/30 text-indigo-100'
                            }
                        `}>
                            {isBotSpeaking ? (
                                <>
                                    <Sparkles size={18} className="animate-spin-slow text-yellow-300" />
                                    <span className="font-semibold tracking-wide">正在讲解...</span>
                                    <div className="h-4 w-[60px] flex items-center">
                                        <AudioVisualizer isActive={true} color="white" width={60} height={16} />
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div className="relative">
                                        <Eye size={18} className="text-indigo-300" />
                                        <span className="absolute -top-1 -right-1 flex h-2 w-2">
                                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                                          <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
                                        </span>
                                    </div>
                                    <span className="text-sm font-medium">正在观察与分析...</span>
                                </>
                            )}
                        </div>
                    </div>
                </>
            )}
            
            {/* Welcome / Disconnected State */}
            {connectionState === ConnectionState.DISCONNECTED && !error && (
                 <div className="absolute inset-0 flex items-center justify-center bg-black/90 z-30">
                    <div className="text-center p-8 max-w-2xl animate-in fade-in zoom-in duration-500">
                        <div className="mb-12">
                            <h1 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-500 mb-6 tracking-tight">
                                未来的学习体验
                            </h1>
                            <p className="text-gray-400 text-lg md:text-xl font-light leading-relaxed">
                                实时连接 AI 导师。使用视频进行互动讲解，为您提供个性化的辅导体验。
                            </p>
                        </div>

                        <div className="flex flex-col items-center gap-8">
                            <button 
                                onClick={startSession}
                                className="px-12 py-4 bg-white text-black hover:bg-gray-100 rounded-full font-bold text-lg transition-all shadow-[0_0_25px_rgba(255,255,255,0.2)] hover:shadow-[0_0_40px_rgba(255,255,255,0.4)] hover:scale-105 active:scale-95 flex items-center gap-3"
                            >
                                立即开始上课
                            </button>
                            
                            <div className="flex items-center gap-3 text-sm text-gray-500 bg-gray-800/50 px-4 py-2 rounded-full border border-gray-700/50 backdrop-blur-sm">
                                <span className="flex items-center gap-2">
                                    <span className="text-xl">{userProfile.avatar}</span>
                                    {userProfile.name ? `欢迎回来，${userProfile.name}` : '欢迎新同学'}
                                </span>
                                <span className="w-1 h-1 bg-gray-600 rounded-full"></span>
                                <button onClick={() => setShowProfileModal(true)} className="text-indigo-400 hover:text-indigo-300 underline transition-colors">
                                    修改资料
                                </button>
                            </div>
                        </div>
                    </div>
                 </div>
            )}

            {/* Error State */}
            {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-40">
                     <div className="text-center p-6 bg-gray-900 border border-red-900/50 rounded-xl max-w-md shadow-2xl">
                        <AlertCircle size={40} className="text-red-500 mx-auto mb-4" />
                        <h3 className="text-xl font-bold text-red-400 mb-2">出错了</h3>
                        <p className="text-gray-300 mb-6">{error}</p>
                        <button 
                            onClick={() => setError(null)}
                            className="px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-colors"
                        >
                            关闭
                        </button>
                     </div>
                </div>
            )}
        </div>

        {/* Controls Bar - Only show when connected */}
        {connectionState !== ConnectionState.DISCONNECTED && (
            <div className="h-24 bg-gray-900 border-t border-gray-800 flex items-center justify-center gap-6 px-4 z-20">
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
                                    : 'bg-gray-800 text-white hover:bg-gray-700 hover:scale-110 active:scale-95'
                                }`}
                                title={isMicMuted ? "取消静音" : "静音"}
                            >
                                {isMicMuted ? <MicOff size={24} /> : <Mic size={24} />}
                            </button>
                        </div>

                        <div className="flex items-center gap-4">
                            <button 
                                onClick={() => saveSessionToHistory(true)}
                                disabled={messages.length === 0}
                                className={`p-4 rounded-full text-white transition-all relative group ${
                                    messages.length === 0 
                                    ? 'bg-gray-800 opacity-50 cursor-not-allowed' 
                                    : 'bg-gray-800 hover:bg-gray-700 hover:scale-110 active:scale-95'
                                }`}
                                title="保存当前对话"
                            >
                                <Save size={24} className={showSaveConfirm ? "text-green-500 transition-colors" : ""} />
                                {showSaveConfirm && (
                                    <span className="absolute -top-10 left-1/2 transform -translate-x-1/2 text-xs font-bold bg-green-500/90 text-white px-3 py-1.5 rounded-lg shadow-lg animate-in fade-in slide-in-from-bottom-2 whitespace-nowrap z-50">
                                        已保存
                                    </span>
                                )}
                            </button>

                            <button 
                                onClick={stopSession}
                                className="p-4 rounded-full bg-red-600 text-white hover:bg-red-500 transition-all shadow-lg hover:shadow-red-600/20 hover:scale-110 active:scale-95"
                                title="结束会话"
                            >
                                <Square size={24} fill="currentColor" />
                            </button>
                        </div>
                    </>
                ) : (
                    <div className="text-sm text-gray-500 italic">
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