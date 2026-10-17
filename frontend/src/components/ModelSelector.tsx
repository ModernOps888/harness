import React from 'react';
import { Box, Layers, Cpu, Sparkles } from 'lucide-react';

export interface ModelOption {
  id: string;
  name: string;
  family: string;
  params: string;
  quant: string;
  context: string;
  mode: '32B-Speculative' | 'Single-GPU' | '70B-LayerStream' | 'Sparse-MoE' | 'Edge-Fast';
  badgeColor: string;
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'qwen2.5-coder:32b',
    name: 'Qwen 2.5 Coder: 32B (+1.5B Draft)',
    family: 'Alibaba Cloud (64L GQA 40:8)',
    params: '32.76B Dense + 1.54B Draft',
    quant: 'Q4_K_M (19.8 GB, 2.99-3.65 tok/s)',
    context: '32k (Q8_0/Q4_0 KV)',
    mode: '32B-Speculative',
    badgeColor: 'border-emerald-500/40 text-emerald-400 bg-emerald-500/10',
  },
  {
    id: 'qwen2.5-coder:14b-instruct-q4_K_S',
    name: 'Qwen 2.5 Coder: 14B Instruct',
    family: 'Alibaba Cloud (48L GQA 40:8)',
    params: '14.7B Dense',
    quant: 'Q4_K_S (8.3 GB, ~8-12 tok/s)',
    context: '32k',
    mode: 'Single-GPU',
    badgeColor: 'border-cyan-500/40 text-cyan-400 bg-cyan-500/10',
  },
  {
    id: 'qwen3-30b-a3b',
    name: 'Qwen3 Coder: 30B-A3B MoE',
    family: 'Alibaba Cloud (128x8 Experts)',
    params: '30.5B Total / 3.3B Active',
    quant: 'Q4_K_M (17.5 GB, ~14-22 tok/s)',
    context: '32k',
    mode: 'Sparse-MoE',
    badgeColor: 'border-amber-500/40 text-amber-400 bg-amber-500/10',
  },
  {
    id: 'llama3.1:70b-instruct-q2_K',
    name: 'Llama 3.1: 70B Instruct',
    family: 'Meta Open-Weight (80L)',
    params: '70.55B Dense',
    quant: 'Q2_K (26.4 GB, 1.38 tok/s)',
    context: '128k',
    mode: '70B-LayerStream',
    badgeColor: 'border-indigo-500/40 text-indigo-400 bg-indigo-500/10',
  },
];

interface Props {
  selectedModel: string;
  onSelectModel: (id: string) => void;
}

export const ModelSelector: React.FC<Props> = ({ selectedModel, onSelectModel }) => {
  return (
    <div className="flex flex-col gap-2 p-4 bg-[#0a0f1d] border-b border-slate-800">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-emerald-400" />
          Active Model Engine
        </span>
        <span className="text-[11px] font-mono text-slate-500">Zero-Copy Mmap DMA Active</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-2.5 mt-1">
        {AVAILABLE_MODELS.map((model) => {
          const isSelected = selectedModel === model.id;
          return (
            <button
              key={model.id}
              onClick={() => onSelectModel(model.id)}
              className={`text-left p-3 rounded-xl border transition-all duration-200 relative overflow-hidden ${
                isSelected
                  ? 'bg-gradient-to-b from-[#131c31] to-[#0f172a] border-emerald-500/60 shadow-lg shadow-emerald-950/40 ring-1 ring-emerald-500/30'
                  : 'bg-[#101726]/60 border-slate-800 hover:border-slate-700 hover:bg-[#131c31]/50'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-slate-200 truncate">{model.name}</span>
                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${model.badgeColor}`}>
                  {model.mode}
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span>{model.family}</span>
                <span className="font-mono text-slate-300">{model.quant}</span>
              </div>
              {isSelected && (
                <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-r from-emerald-500 via-cyan-400 to-indigo-500" />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};
