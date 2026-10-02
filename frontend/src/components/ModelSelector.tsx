import React from 'react';
import { Box, Layers, Cpu, Sparkles } from 'lucide-react';

export interface ModelOption {
  id: string;
  name: string;
  family: string;
  params: string;
  quant: string;
  context: string;
  mode: 'Single-GPU' | '70B-LayerStream' | 'Sparse-MoE' | 'Edge-Fast';
  badgeColor: string;
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'llama3.1:70b-instruct-q2_K',
    name: 'Llama 3.1: 70B Instruct',
    family: 'Meta Open-Weight',
    params: '70.55B Dense (1.05 tok/s)',
    quant: 'Q2_K (26.4 GB disk)',
    context: '128k',
    mode: '70B-LayerStream',
    badgeColor: 'border-cyan-500/40 text-cyan-400 bg-cyan-500/10',
  },
  {
    id: 'qwen2.5-coder:7b',
    name: 'Qwen 2.5 Coder: 7B',
    family: 'Alibaba Cloud',
    params: '7.61B Dense (79.4 tok/s)',
    quant: 'Q4_K_M (4.7 GB VRAM)',
    context: '32k',
    mode: 'Single-GPU',
    badgeColor: 'border-emerald-500/40 text-emerald-400 bg-emerald-500/10',
  },
  {
    id: 'llama3.2:1b',
    name: 'Llama 3.2: 1B Instruct',
    family: 'Meta Open-Weight',
    params: '1.24B Dense (120+ tok/s)',
    quant: 'Q8_0 (1.3 GB VRAM)',
    context: '8k',
    mode: 'Edge-Fast',
    badgeColor: 'border-amber-500/40 text-amber-400 bg-amber-500/10',
  },
  {
    id: 'qwen3:8b',
    name: 'Qwen: 8B Instruct',
    family: 'Alibaba Cloud',
    params: '8.0B Dense (75 tok/s)',
    quant: 'Q4_K_M (5.2 GB VRAM)',
    context: '32k',
    mode: 'Single-GPU',
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
