import React, { useEffect, useRef, useState } from 'react';
import { ChatMessage, SavedSession, UserProfile } from '../types';
import { User, Bot, History, ChevronLeft, Trash2, MessageSquare, Calendar, BookOpen, Lightbulb } from 'lucide-react';

interface TranscriptProps {
  messages: ChatMessage[];
  userProfile: UserProfile;
}

type ViewMode = 'live' | 'history_list' | 'history_detail';

const Transcript: React.FC<TranscriptProps> = ({ messages, userProfile }) => {
  const bottomRef = useRef<HTMLDivElement>(null);
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
              {msg.role === 'user' ? (
                userProfile.avatar ? (
                  <span className="text-lg leading-none select-none" role="img" aria-label="avatar">
                    {userProfile.avatar}
                  </span>
                ) : (
                  <User size={16} />
                )
              ) : (
                <Bot size={16} />
              )}
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
                            {session.summary?.overview 
                                ? <span className="text-emerald-300 font-medium">【总结】{session.summary.overview.slice(0, 30)}...</span>
                                : (session.preview || "（无文本内容）")
                            }
                        </p>
                        <div className="mt-3 flex items-center justify-between">
                            <div className="flex items-center gap-2 text-xs text-gray-500">
                                <MessageSquare size={12} />
                                <span>{session.messages.length} 条对话</span>
                            </div>
                            {session.summary && (
                                <div className="flex items-center gap-1 text-xs text-emerald-500/80 bg-emerald-500/10 px-2 py-0.5 rounded">
                                    <BookOpen size={12} />
                                    <span>含学习报告</span>
                                </div>
                            )}
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
               
               {/* Summary Card in History View */}
               {selectedSession.summary && (
                   <div className="m-4 p-4 bg-gradient-to-br from-gray-800 to-gray-900 border border-indigo-500/20 rounded-xl shadow-lg">
                       <h3 className="text-sm font-bold text-indigo-300 mb-3 flex items-center gap-2">
                           <BookOpen size={16} /> 学习报告
                       </h3>
                       
                       <div className="mb-3">
                           <h4 className="text-xs font-semibold text-gray-400 mb-1 uppercase">概览</h4>
                           <p className="text-sm text-gray-300 leading-relaxed">{selectedSession.summary.overview}</p>
                       </div>
                       
                       <div>
                           <h4 className="text-xs font-semibold text-gray-400 mb-1 uppercase">知识点</h4>
                           <ul className="space-y-1">
                               {selectedSession.summary.knowledgePoints.map((point, i) => (
                                   <li key={i} className="flex gap-2 items-start text-sm text-gray-300">
                                       <Lightbulb size={14} className="text-emerald-500 mt-0.5 flex-shrink-0" />
                                       <span>{point}</span>
                                   </li>
                               ))}
                           </ul>
                       </div>
                   </div>
               )}

               <MessageList msgs={selectedSession.messages} />
          </div>
      )}

    </div>
  );
};

export default Transcript;