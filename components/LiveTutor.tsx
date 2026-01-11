import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { Video, Mic, MicOff, Play, Square, AlertCircle, Volume2, Sparkles, Eye, Settings, VolumeX, RefreshCw, Camera, FlipHorizontal, Lightbulb, Key, X, MessageCircleQuestion, ArrowRight, ScanEye, Target, UserRoundPen, Check, ChevronRight } from 'lucide-react';
import { ConnectionState, ChatMessage, SavedSession, UserProfile } from '../types';
import { createPcmBlob, decode, decodeAudioData, blobToBase64 } from '../utils/audioUtils';
import AudioVisualizer from './AudioVisualizer';
import Transcript from './Transcript';

// Configuration constants
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-12-2025';

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

8. **Text Input Handling**: If the user sends a text message or question (e.g. via the chat input), YOU MUST RESPOND TO IT IMMEDIATELY. Do not ignore text input. You can combine text questions with visual context if relevant.
`;

const AVATAR_OPTIONS = ['🎓', '🚀', '🌟', '🐶', '🐱', '🦊', '🐯', '🐼', '🧠', '💡'];
const FRAME_RATE = 2; // Frames per second sent to model (0.5s interval)
const PCM_SAMPLE_RATE = 16000; // Input sample rate
const OUTPUT_SAMPLE_RATE = 24000; // Output sample rate

const LiveTutor: React.FC = () => {
  // --- State ---
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Profile State
  const [userProfile, setUserProfile] = useState<UserProfile>({ name: '', age: '', avatar: '🎓' });
  const [showProfileModal, setShowProfileModal] = useState(false);

  // Insight State
  const [insightData, setInsightData] = useState<{ knowledge: string | null; eye: string | null }>({ knowledge: null, eye: null });
  const [activePopup, setActivePopup] = useState<{ type: 'knowledge' | 'eye'; content: string } | null>(null);
  const [showBlurWarning, setShowBlurWarning] = useState(false);
  const [scannerActive, setScannerActive] = useState(false);

  // Media State
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [isSpeakerMuted, setIsSpeakerMuted] = useState(false);
  const [isVideoMirrored, setIsVideoMirrored] = useState(true); // Default mirror for selfie feel
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [showSettings, setShowSettings] = useState(false);

  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [inputAnalyser, setInputAnalyser] = useState<AnalyserNode | null>(null);

  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]); 
  const hasSavedRef = useRef<boolean>(false);
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

  // Load Profile on Mount
  useEffect(() => {
      try {
          const savedProfile = localStorage.getItem('user_profile');
          if (savedProfile) {
              setUserProfile(JSON.parse(savedProfile));
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
        // Keywords: finger, pointing, pointing at, pen, see your finger, etc.
        const gestureKeywords = /(?:手指|指着|指向|看这里|pointing at|your finger|this area|circled|highlighted|笔)/i;
        if (gestureKeywords.test(text)) {
             setScannerActive(true);
             if (scannerTimeoutRef.current) clearTimeout(scannerTimeoutRef.current);
             scannerTimeoutRef.current = window.setTimeout(() => setScannerActive(false), 5000);
        }

        // 3. Detect Blurry/Adjustment requests
        // Keywords: blurry, unclear, can't see, adjust camera, focus
        const blurKeywords = /(?:看不清|模糊|调整.*摄像头|太远|太小|拿近|unclear|blurry|too far|adjust.*camera)/i;
        if (blurKeywords.test(text)) {
            setShowBlurWarning(true);
            
            // Clear previous timeout if exists to reset the timer
            if (blurTimeoutRef.current) {
                clearTimeout(blurTimeoutRef.current);
            }
            
            // Auto hide after 6 seconds
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
            // Prefer back camera if available on mobile, else first one
            const backCamera = cameras.find(c => c.label.toLowerCase().includes('back') || c.label.toLowerCase().includes('environment'));
            setSelectedCameraId(backCamera ? backCamera.deviceId : cameras[0].deviceId);
            // If back camera, default to NOT mirrored
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
        // Smooth transition
        const currentTime = outputContextRef.current?.currentTime || 0;
        outputNodeRef.current.gain.cancelScheduledValues(currentTime);
        outputNodeRef.current.gain.setTargetAtTime(isSpeakerMuted ? 0 : 1, currentTime, 0.1);
    }
  }, [isSpeakerMuted]);

  // Handle Camera Switch during active session
  const switchCamera = async (deviceId: string) => {
    setSelectedCameraId(deviceId);
    
    // If we have an active video element, restart the stream
    if (videoRef.current && connectionState === ConnectionState.CONNECTED) {
        try {
            // Stop old tracks
            const oldStream = videoRef.current.srcObject as MediaStream;
            if (oldStream) {
                oldStream.getVideoTracks().forEach(t => t.stop());
            }

            // Start new stream
            const newStream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: deviceId }, width: 1280, height: 720 },
                audio: false // Audio is handled separately
            });
            
            videoRef.current.srcObject = newStream;
            await videoRef.current.play();
            
            // Re-bind audio stream to input context if needed (usually audio device doesn't change with camera)
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

  // --- Save Session Logic ---
  const saveSessionToHistory = useCallback(() => {
    if (messagesRef.current.length === 0 || hasSavedRef.current) return;
    try {
        const historyData = localStorage.getItem('tutoring_history');
        const history: SavedSession[] = historyData ? JSON.parse(historyData) : [];
        const firstUserMsg = messagesRef.current.find(m => m.role === 'user');
        const preview = firstUserMsg 
            ? (firstUserMsg.text.slice(0, 40) + (firstUserMsg.text.length > 40 ? '...' : ''))
            : '无内容会话';
        const newSession: SavedSession = {
            id: Date.now().toString(),
            timestamp: Date.now(),
            preview,
            messages: messagesRef.current
        };
        const updatedHistory = [newSession, ...history];
        localStorage.setItem('tutoring_history', JSON.stringify(updatedHistory));
        hasSavedRef.current = true;
    } catch (e) {
        console.error("Failed to save session history", e);
    }
  }, []);

  // --- Cleanup Function ---
  const stopSession = useCallback(async () => {
    saveSessionToHistory();
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
  }, [saveSessionToHistory]);

  // --- Start Function ---
  const startSession = async () => {
    try {
      setConnectionState(ConnectionState.CONNECTING);
      setError(null);
      setMessages([]);
      setInsightData({ knowledge: null, eye: null }); // Clear insights on start
      setActivePopup(null);
      setShowBlurWarning(false);
      setScannerActive(false);
      hasSavedRef.current = false;

      // 1. Setup Camera (Use selected ID)
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
      
      // Gain Node for Speaker Mute
      outputNodeRef.current = outputContextRef.current.createGain();
      outputNodeRef.current.gain.value = isSpeakerMuted ? 0 : 1;
      outputNodeRef.current.connect(outputContextRef.current.destination);
      
      nextStartTimeRef.current = 0;

      // 4. Construct System Instruction with Profile
      let currentSystemInstruction = BASE_SYSTEM_INSTRUCTION;
      if (userProfile.name) {
          currentSystemInstruction += `\n9. **Student Profile**: The student's name is "${userProfile.name}". Use their name occasionally to be friendly.`;
      }
      if (userProfile.age) {
          currentSystemInstruction += `\n10. **Age Appropriateness**: The student is ${userProfile.age} years old. ADJUST YOUR EXPLANATION COMPLEXITY AND TONE TO MATCH A ${userProfile.age}-YEAR-OLD CHILD.`;
      } else {
          currentSystemInstruction += `\n10. **Age Appropriateness**: Assume the student is a middle school student. Explain concepts clearly and simply.`;
      }

      // 5. Connect to Gemini Live
      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: currentSystemInstruction,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
        },
        callbacks: {
          onopen: () => {
            console.log('Gemini Live Connection Opened');
            setConnectionState(ConnectionState.CONNECTED);
            
            // Start Audio Streaming (Microphone)
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

            // Start Video Streaming
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
            setError("连接发生错误，请重试。");
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

  const startVideoStreaming = (sessionPromise: Promise<any>) => {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    frameIntervalRef.current = window.setInterval(() => {
        if (!videoRef.current || !canvasRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx || video.readyState < 2) return;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        // Important: Draw the video exactly as is (raw). 
        // We do NOT apply the mirror transform to the canvas sent to AI, only to the UI video element.
        ctx.drawImage(video, 0, 0);
        
        canvas.toBlob(async (blob) => {
            if (blob) {
                const base64 = await blobToBase64(blob);
                sessionPromise.then(session => {
                    session.sendRealtimeInput({ media: { mimeType: 'image/jpeg', data: base64 } });
                });
            }
        }, 'image/jpeg', 0.6);
    }, 1000 / FRAME_RATE);
  };

  const handleSendMessage = useCallback((text: string) => {
    if (!sessionPromiseRef.current) return;
    updateTranscript('user', text, true);
    sessionPromiseRef.current.then(session => {
        // Safe check for send method. If unavailable, we prevent crash.
        // The error "session.send is not a function" typically means the method doesn't exist on this SDK version/build.
        // We use 'as any' to bypass TS check and runtime check to avoid crash.
        const s = session as any;
        if (typeof s.send === 'function') {
            s.send({
                 clientContent: {
                     turns: [{ role: 'user', parts: [{ text }] }],
                     turnComplete: true
                 }
            });
        } else {
            console.warn("session.send is not available in this SDK version. Text message not sent to model.");
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
        const base64 = await blobToBase64(file);
        const mimeType = file.type;
        const msgText = `[用户上传了文件: ${file.name}]`;
        
        updateTranscript('user', msgText, true);

        sessionPromiseRef.current.then(session => {
             // Use sendRealtimeInput for image data - this is the supported method for Live API media
             session.sendRealtimeInput({ 
                 media: { mimeType, data: base64 } 
             });
        });
    } catch (e) {
        console.error("File upload failed", e);
        alert("文件处理失败");
    }
  }, [updateTranscript]);

  const handleAskExplain = () => {
      if (!activePopup) return;
      const prompt = `请详细为我讲解一下这个${activePopup.type === 'knowledge' ? '知识点' : '题眼'}：${activePopup.content}`;
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
        {/* Header */}
        <div className="absolute top-0 left-0 right-0 z-10 p-4 bg-gradient-to-b from-black/70 to-transparent flex justify-between items-center">
            <div className="flex items-center gap-2">
                <div className="bg-indigo-600 p-2 rounded-lg">
                    <Video size={20} className="text-white" />
                </div>
                <h1 className="text-xl font-bold tracking-tight">AI 实时解题导师</h1>
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
                            <div className="text-xs text-indigo-400 mt-2 flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                                点击查看详情 <ArrowRight size={12} className="ml-1" />
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
                            <div className="text-xs text-emerald-400 mt-2 flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                                点击查看详情 <ArrowRight size={12} className="ml-1" />
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

            {/* Settings & Profile Overlay (Top Right inside video) */}
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
                        <div className="relative">
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
                        
                        <div className="h-px bg-gray-700/50 my-1" />
                        
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

            {/* Profile Edit Modal */}
            {showProfileModal && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
                    <div className="bg-gray-900 border border-gray-700 w-full max-w-sm rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
                        <div className="p-4 border-b border-gray-700/50 flex justify-between items-center bg-gray-800/50">
                            <h3 className="font-bold text-lg text-white flex items-center gap-2">
                                <UserRoundPen size={20} className="text-indigo-400" />
                                个人资料设置
                            </h3>
                            <button onClick={() => setShowProfileModal(false)} className="p-1 hover:bg-white/10 rounded-full text-gray-400 hover:text-white transition-colors">
                                <X size={20} />
                            </button>
                        </div>
                        
                        <div className="p-6 space-y-6">
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
                        </div>

                        <div className="p-4 bg-gray-800/30 border-t border-gray-700/50">
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
                 <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-30">
                    <div className="text-center p-8">
                        <div className="inline-block p-4 rounded-full bg-indigo-600/20 mb-4 animate-bounce">
                            <Video size={48} className="text-indigo-400" />
                        </div>
                        <h2 className="text-2xl font-bold mb-2">准备好开始了吗？</h2>
                        <div className="flex items-center justify-center gap-2 mb-8">
                            <span className="text-2xl">{userProfile.avatar}</span>
                            <p className="text-gray-300">
                                欢迎回来{userProfile.name ? `，${userProfile.name}` : ''}
                            </p>
                            <button onClick={() => setShowProfileModal(true)} className="text-indigo-400 hover:text-indigo-300 text-xs underline">修改资料</button>
                        </div>
                        
                        <p className="text-gray-400 max-w-md mx-auto mb-8">
                            请将摄像头对准题目，并用手指或笔指向你想问的具体位置。
                        </p>
                        <button 
                            onClick={startSession}
                            className="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-full font-semibold transition-all shadow-lg shadow-indigo-600/20 flex items-center gap-2 mx-auto hover:scale-105 active:scale-95"
                        >
                            <Play size={20} fill="currentColor" />
                            开始辅导
                        </button>
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

        {/* Controls Bar */}
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

                    <button 
                        onClick={stopSession}
                        className="p-4 rounded-full bg-red-600 text-white hover:bg-red-500 transition-all shadow-lg hover:shadow-red-600/20 hover:scale-110 active:scale-95"
                        title="结束会话"
                    >
                        <Square size={24} fill="currentColor" />
                    </button>
                </>
            ) : (
                <div className="text-sm text-gray-500 italic">
                    {connectionState === ConnectionState.CONNECTING ? "正在建立连接..." : "等待开始..."}
                </div>
            )}
        </div>
      </div>

      {/* Sidebar: Transcript with Input */}
      <Transcript 
        messages={messages} 
        onSendMessage={handleSendMessage}
        onSendFile={handleSendFile}
        disabled={connectionState !== ConnectionState.CONNECTED}
      />
    </div>
  );
};

export default LiveTutor;