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
    id: 'Qwen3.8-27B-ISQ',
    name: 'Qwen 3.8: 27B ISQ',
    family: 'Alibaba Cloud',
    params: '27B Dense',
    quant: 'ISQ (Q4_K_M + FP8)',
    context: '131k',
    mode: 'Single-GPU',
    badgeColor: 'border-emerald-500/40 text-emerald-400 bg-emerald-500/10',
  },
  {
    id: 'Llama-4-Scout-70B-LayerStream',
    name: 'Llama 4 Scout: 70B Stream',
    family: 'Meta Open-Weight',
    params: '70B MoE/Dense',
    quant: 'Q4_K_M Layer-Stream',
    context: '128k',
    mode: '70B-LayerStream',
    badgeColor: 'border-cyan-500/40 text-cyan-400 bg-cyan-500/10',
  },
  {
    id: 'Qwen-2.5-72B-Instruct',
    name: 'Qwen 2.5: 72B Instruct',
    family: 'Alibaba Cloud',
    params: '72B Dense (UMA 128GB)',
    quant: 'Q4_K_M / FP8',
    context: '131k',
    mode: '70B-LayerStream',
    badgeColor: 'border-teal-500/40 text-teal-400 bg-teal-500/10',
  },
  {
    id: 'DeepSeek-R1-671B-SparseMoE',
    name: 'DeepSeek R1: 671B Sparse MoE',
    family: 'DeepSeek AI',
    params: '671B (37B Active)',
    quant: 'FP8 / 2-bit Ternary',
    context: '163k',
    mode: 'Sparse-MoE',
    badgeColor: 'border-purple-500/40 text-purple-400 bg-purple-500/10',
  },
  {
    id: 'DeepSeek-V4.1-Flash-MoE',
    name: 'DeepSeek V4.1 Flash',
    family: 'DeepSeek AI',
    params: '236B (21B Active)',
    quant: 'FP8 E4M3',
    context: '1,048k',
    mode: 'Sparse-MoE',
    badgeColor: 'border-indigo-500/40 text-indigo-400 bg-indigo-500/10',
  },
  {
    id: 'Phi-4-Reasoning-14B',
    name: 'Phi-4 Reasoning',
    family: 'Microsoft',
    params: '14B Dense',
    quant: 'NF4 Optimal',
    context: '64k',
    mode: 'Edge-Fast',
    badgeColor: 'border-amber-500/40 text-amber-400 bg-amber-500/10',
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
