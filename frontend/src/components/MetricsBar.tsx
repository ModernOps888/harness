import React from 'react';
import { Zap, Cpu, Database, Activity, ShieldCheck, Sparkles, Brain, Bug } from 'lucide-react';

interface MetricsProps {
  tokPerSec: number;
  ttftMs: number;
  vramUsedMb: number;
  vramTotalMb: number;
  kvFragmentation: number;
  speculativeAcceptance: number;
  confidenceScore: number;
  isStreaming: boolean;
  spikingSparsity: number;
}

export const MetricsBar: React.FC<MetricsProps> = ({
  tokPerSec,
  ttftMs,
  vramUsedMb,
  vramTotalMb,
  kvFragmentation,
  speculativeAcceptance,
  confidenceScore,
  isStreaming,
  spikingSparsity,
}) => {
  const vramPct = Math.round((vramUsedMb / Math.max(vramTotalMb, 1)) * 100);

  return (
    <div className="grid grid-cols-2 md:grid-cols-7 gap-3 p-4 bg-[#0d121f] border-b border-slate-800">
      {/* tok/s Odometer */}
      <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1.5 font-medium">
            <Zap className={`w-3.5 h-3.5 ${isStreaming ? 'text-amber-400 animate-pulse' : 'text-slate-400'}`} />
            Throughput
          </span>
          {isStreaming && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />}
        </div>
        <div className="mt-2 flex items-baseline gap-1">
          <span className="text-2xl font-bold font-mono text-emerald-400">{tokPerSec.toFixed(1)}</span>
          <span className="text-xs text-slate-500 font-mono">tok/s</span>
        </div>
      </div>

      {/* TTFT Latency */}
      <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center text-xs text-slate-400">
          <span className="flex items-center gap-1.5 font-medium">
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            TTFT Latency
          </span>
        </div>
        <div className="mt-2 flex items-baseline gap-1">
          <span className="text-2xl font-bold font-mono text-cyan-300">{ttftMs.toFixed(1)}</span>
          <span className="text-xs text-slate-500 font-mono">ms</span>
        </div>
      </div>

      {/* VRAM Allocation */}
      <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1.5 font-medium">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            VRAM Usage
          </span>
          <span className="text-[11px] font-mono text-slate-400">{vramPct}%</span>
        </div>
        <div className="mt-2">
          <div className="text-lg font-bold font-mono text-slate-200">
            {(vramUsedMb / 1024).toFixed(1)} <span className="text-xs text-slate-500">/ {(vramTotalMb / 1024).toFixed(0)} GB</span>
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full mt-1.5 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(vramPct, 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* PagedAttention KV Cache */}
      <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1.5 font-medium">
            <Database className="w-3.5 h-3.5 text-purple-400" />
            KV Waste (Paged)
          </span>
        </div>
        <div className="mt-2 flex items-baseline gap-1">
          <span className="text-2xl font-bold font-mono text-purple-300">{kvFragmentation.toFixed(1)}%</span>
          <span className="text-xs text-slate-500 font-mono">fragment</span>
        </div>
      </div>

      {/* Spiking Neural Sparsity */}
      <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1.5 font-medium">
            <Brain className="w-3.5 h-3.5 text-rose-400" />
            LIF Spike Sparsity
          </span>
          <span className="text-[10px] bg-rose-500/10 text-rose-300 px-1.5 py-0.5 rounded border border-rose-500/20">Bio-SNN</span>
        </div>
        <div className="mt-2 flex items-baseline gap-1">
          <span className="text-2xl font-bold font-mono text-rose-300">{(spikingSparsity * 100).toFixed(0)}%</span>
          <span className="text-xs text-slate-500 font-mono">FLOP saved</span>
        </div>
      </div>

      {/* Speculative Decoding */}
      <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1.5 font-medium">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            EAGLE Draft Accept
          </span>
        </div>
        <div className="mt-2 flex items-baseline gap-1">
          <span className="text-2xl font-bold font-mono text-amber-300">{(speculativeAcceptance * 100).toFixed(0)}%</span>
          <span className="text-xs text-slate-500 font-mono">verified</span>
        </div>
      </div>

      {/* Anti-Hallucination Confidence */}
      <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1.5 font-medium">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            Confidence
          </span>
          <span className="text-[10px] bg-emerald-500/10 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-500/20">Calibrated</span>
        </div>
        <div className="mt-2 flex items-baseline gap-1">
          <span className="text-2xl font-bold font-mono text-emerald-300">{(confidenceScore * 100).toFixed(0)}%</span>
          <span className="text-xs text-slate-500 font-mono">factual</span>
        </div>
      </div>
    </div>
  );
};
