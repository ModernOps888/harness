import React, { useState, useRef, useEffect } from 'react';
import { Send, Square, Bot, User, ShieldCheck, Zap, Copy, Check } from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  tokPerSec?: number;
  confidenceScore?: number;
  citations?: string[];
}

interface Props {
  messages: ChatMessage[];
  isStreaming: boolean;
  onSendMessage: (text: string) => void;
  onAbort: () => void;
}

export const ChatView: React.FC<Props> = ({
  messages,
  isStreaming,
  onSendMessage,
  onAbort,
}) => {
  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="flex flex-col flex-1 h-full min-h-[500px] bg-[#080b11]">
      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-20">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mb-4">
              <Bot className="w-6 h-6 text-emerald-400" />
            </div>
            <h2 className="text-lg font-bold text-slate-200">HARNESS Pure-Rust Engine Ready</h2>
            <p className="text-xs text-slate-400 max-w-md mt-1 mb-6">
              World-class local LLM inference with zero-copy memory mapping, PagedAttention, speculative decoding, and calibrated anti-hallucination.
            </p>
            <div className="flex flex-wrap gap-2 justify-center max-w-lg">
              {[
                'Benchmark 70B layer streaming speed',
                'Demonstrate strict JSON schema constrained decoding',
                'Explain how PagedAttention reduces KV waste to <2%',
                'Test real-time Shannon entropy hallucination detection',
              ].map((pill, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendMessage(pill)}
                  className="text-xs bg-[#111827] hover:bg-[#162035] text-slate-300 hover:text-emerald-300 border border-slate-800 hover:border-emerald-500/40 px-3 py-1.5 rounded-lg transition-all"
                >
                  {pill}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => {
            const isUser = msg.role === 'user';
            return (
              <div
                key={msg.id}
                className={`flex gap-3 max-w-3xl ${isUser ? 'ml-auto' : 'mr-auto'} w-full`}
              >
                <div
                  className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
                    isUser
                      ? 'bg-blue-600/20 border border-blue-500/30 text-blue-400'
                      : 'bg-emerald-600/20 border border-emerald-500/30 text-emerald-400'
                  }`}
                >
                  {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>

                <div className="flex-1 space-y-1.5">
                  <div className="flex items-center justify-between text-[11px] text-slate-500">
                    <span className="font-semibold text-slate-400">
                      {isUser ? 'You' : 'HARNESS Assistant'}
                    </span>
                    {!isUser && (
                      <div className="flex items-center gap-2">
                        {msg.tokPerSec && (
                          <span className="font-mono text-emerald-400 flex items-center gap-1">
                            <Zap className="w-3 h-3" />
                            {msg.tokPerSec.toFixed(1)} tok/s
                          </span>
                        )}
                        {msg.confidenceScore && (
                          <span className="font-mono text-cyan-400 flex items-center gap-1">
                            <ShieldCheck className="w-3 h-3" />
                            {(msg.confidenceScore * 100).toFixed(0)}% factual
                          </span>
                        )}
                        <button
                          onClick={() => handleCopy(msg.id, msg.content)}
                          className="hover:text-slate-300 transition-colors"
                        >
                          {copiedId === msg.id ? (
                            <Check className="w-3 h-3 text-emerald-400" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                        </button>
                      </div>
                    )}
                  </div>

                  <div
                    className={`p-4 rounded-2xl text-sm leading-relaxed ${
                      isUser
                        ? 'bg-blue-600/10 border border-blue-500/20 text-blue-100 rounded-tr-sm'
                        : 'bg-[#0f1524] border border-slate-800 text-slate-200 rounded-tl-sm shadow-lg'
                    }`}
                  >
                    {isUser ? (
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    ) : (
                      <MarkdownRenderer content={msg.content} />
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-slate-800 bg-[#0d121f]">
        <div className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message or instruction for HARNESS..."
            className="w-full bg-[#111827] border border-slate-700/80 rounded-xl pl-4 pr-24 py-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/70 focus:ring-1 focus:ring-emerald-500/30 transition-all font-sans"
          />
          <div className="absolute right-2 flex items-center gap-1.5">
            {isStreaming ? (
              <button
                type="button"
                onClick={onAbort}
                className="bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/40 p-2 rounded-lg transition-all"
                title="Abort inference"
              >
                <Square className="w-4 h-4 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 disabled:hover:bg-emerald-500 text-slate-950 p-2 rounded-lg font-bold transition-all"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
};
