import React, { useEffect, useRef, useState } from 'react';
import { ChatMessage, SavedSession } from '../types';
import { User, Bot, Send, History, ChevronLeft, Trash2, MessageSquare, Calendar, Paperclip, FileText } from 'lucide-react';

interface TranscriptProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  onSendFile: (file: File) => void;
  disabled: boolean;
}

type ViewMode = 'live' | 'history_list' | 'history_detail';

const Transcript: React.FC<TranscriptProps> = ({ messages, onSendMessage, onSendFile, disabled }) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [input, setInput] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('live');
  const [savedSessions, setSavedSessions] = useState<SavedSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<SavedSession | null>(null);

  useEffect(() => {
    if (viewMode === 'live') {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, viewMode]);

  const loadHistory = () => {
    try {
      const historyData = localStorage.getItem('tutoring_history');
      if (historyData) {
        setSavedSessions(JSON.parse(historyData));
      } else {
        setSavedSessions([]);
      }
    } catch (e) {
      console.error("Failed to load history", e);
    }
  };

  const handleToggleHistory = () => {
    if (viewMode === 'live') {
      loadHistory();
      setViewMode('history_list');
    } else {
      setViewMode('live');
      setSelectedSession(null);
    }
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm('确定要删除这条记录吗？')) {
      const updated = savedSessions.filter(s => s.id !== id);
      setSavedSessions(updated);
      localStorage.setItem('tutoring_history', JSON.stringify(updated));
      
      if (selectedSession?.id === id) {
        setViewMode('history_list');
        setSelectedSession(null);
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
        onSendFile(file);
    }
    // Reset input so the same file can be selected again if needed
    if (fileInputRef.current) {
        fileInputRef.current.value = '';
    }
  };

  // --- Render Components ---

  const MessageList = ({ msgs }: { msgs: ChatMessage[] }) => (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {msgs.length === 0 ? (
         <div className="text-gray-500 text-center mt-10 text-sm">
           <p>暂无消息内容。</p>
         </div>
      ) : (
        msgs.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 ${
              msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'
            }`}
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                msg.role === 'user' ? 'bg-indigo-600' : 'bg-emerald-600'
              }`}
            >
              {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
            </div>
            <div
              className={`p-3 rounded-lg text-sm max-w-[85%] ${
                msg.role === 'user'
                  ? 'bg-indigo-900/50 text-indigo-100'
                  : 'bg-gray-800 text-gray-100'
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.text}</p>
              {!msg.isComplete && (
                <span className="inline-block w-1.5 h-1.5 bg-gray-400 rounded-full animate-pulse ml-1" />
              )}
            </div>
          </div>
        ))
      )}
      <div ref={bottomRef} />
    </div>
  );

  return (
    <div className="flex flex-col h-full bg-gray-900 border-l border-gray-700 overflow-hidden w-80 md:w-96">
      
      {/* Header */}
      <div className="p-4 border-b border-gray-700 bg-gray-800 flex justify-between items-center shadow-md z-10">
        <div className="flex items-center gap-2">
            {viewMode === 'history_detail' ? (
                <button 
                  onClick={() => setViewMode('history_list')}
                  className="p-1 hover:bg-gray-700 rounded-full transition-colors"
                >
                    <ChevronLeft size={20} />
                </button>
            ) : (
                <MessageSquare size={18} className="text-indigo-400" />
            )}
            <h2 className="font-semibold text-lg text-white">
                {viewMode === 'live' ? '实时对话' : 
                 viewMode === 'history_list' ? '历史记录' : '对话回顾'}
            </h2>
        </div>
        
        {viewMode !== 'history_detail' && (
            <button 
                onClick={handleToggleHistory}
                className={`p-2 rounded-lg transition-colors ${
                    viewMode === 'live' 
                    ? 'hover:bg-gray-700 text-gray-400 hover:text-white' 
                    : 'bg-indigo-600 text-white hover:bg-indigo-500'
                }`}
                title={viewMode === 'live' ? "查看历史" : "返回实时"}
            >
                {viewMode === 'live' ? <History size={18} /> : <MessageSquare size={18} />}
            </button>
        )}
      </div>

      {/* Content Area */}
      {viewMode === 'live' && (
        <>
            <div className="flex-1 overflow-y-auto relative">
                {messages.length === 0 && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 p-6 text-center">
                        <MessageSquare size={48} className="mb-4 opacity-20" />
                        <p className="mb-2">尚未开始对话。</p>
                        <p className="text-xs text-gray-600">点击屏幕下方的“开始辅导”按钮，并对准题目即可开始。</p>
                    </div>
                )}
                <MessageList msgs={messages} />
            </div>

            <div className="p-4 border-t border-gray-800 bg-gray-900">
                <form onSubmit={handleSubmit} className="flex gap-2 items-center">
                    {/* File Input (Hidden) */}
                    <input 
                        type="file" 
                        ref={fileInputRef} 
                        className="hidden" 
                        accept="image/*,application/pdf"
                        onChange={handleFileSelect}
                        disabled={disabled}
                    />
                    
                    {/* File Button */}
                    <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={disabled}
                        className="p-3 bg-gray-800 text-gray-400 rounded-full hover:bg-gray-700 hover:text-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        title="上传图片或PDF"
                    >
                        <Paperclip size={18} />
                    </button>

                    <div className="relative flex-1">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            disabled={disabled}
                            placeholder={disabled ? "请先连接..." : "输入消息..."}
                            className="w-full bg-gray-800 text-white placeholder-gray-500 rounded-full py-3 pl-4 pr-12 focus:outline-none focus:ring-2 focus:ring-indigo-500 border border-gray-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm transition-all"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || disabled}
                            className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 bg-indigo-600 text-white rounded-full hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 transition-colors"
                        >
                            <Send size={16} />
                        </button>
                    </div>
                </form>
            </div>
        </>
      )}

      {viewMode === 'history_list' && (
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {savedSessions.length === 0 ? (
                <div className="text-gray-500 text-center mt-20 text-sm">
                    <History size={48} className="mx-auto mb-4 opacity-20" />
                    <p>暂无历史记录</p>
                </div>
            ) : (
                savedSessions.map(session => (
                    <div 
                        key={session.id}
                        onClick={() => {
                            setSelectedSession(session);
                            setViewMode('history_detail');
                        }}
                        className="bg-gray-800 border border-gray-700 rounded-xl p-4 cursor-pointer hover:bg-gray-750 hover:border-indigo-500/50 transition-all group relative"
                    >
                        <div className="flex justify-between items-start mb-2">
                             <div className="flex items-center gap-2 text-xs text-indigo-300 bg-indigo-900/30 px-2 py-1 rounded">
                                <Calendar size={12} />
                                <span>{new Date(session.timestamp).toLocaleString('zh-CN')}</span>
                             </div>
                             <button 
                                onClick={(e) => handleDeleteSession(session.id, e)}
                                className="text-gray-600 hover:text-red-400 p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                                title="删除记录"
                             >
                                <Trash2 size={16} />
                             </button>
                        </div>
                        <p className="text-sm text-gray-300 line-clamp-2">
                            {session.preview || "（无文本内容）"}
                        </p>
                        <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
                             <MessageSquare size={12} />
                             <span>{session.messages.length} 条对话</span>
                        </div>
                    </div>
                ))
            )}
        </div>
      )}

      {viewMode === 'history_detail' && selectedSession && (
          <div className="flex-1 overflow-y-auto flex flex-col">
               <div className="bg-gray-800/50 p-2 text-center text-xs text-gray-400 border-b border-gray-700/50">
                  {new Date(selectedSession.timestamp).toLocaleString('zh-CN')}
               </div>
               <MessageList msgs={selectedSession.messages} />
          </div>
      )}

    </div>
  );
};

export default Transcript;