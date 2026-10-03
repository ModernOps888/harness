import React from 'react';
import { Zap, Cpu, Database, Activity, ShieldCheck, Sparkles, Brain, Server, HardDrive } from 'lucide-react';

export interface HardwareProfile {
  os: string;
  arch: string;
  host_ram_gb: number;
  accelerator_name: string;
  vram_gb: number;
  is_unified_memory: boolean;
  memory_bandwidth_gbps: number;
  recommended_70b_strategy: string;
  max_supported_context: number;
}

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
  hardware?: HardwareProfile | null;
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
  hardware,
}) => {
  const vramPct = Math.round((vramUsedMb / Math.max(vramTotalMb, 1)) * 100);

  return (
    <div className="bg-[#0d121f] border-b border-slate-800">
      {/* Live Hardware Interconnect & Bandwidth Telemetry Strip */}
      <div className="px-4 py-2 bg-[#090d16] border-b border-slate-800/60 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-3">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 font-semibold text-slate-200">
            <Server className="w-3.5 h-3.5 text-cyan-400" />
            <span>Platform:</span>
            <span className="text-cyan-300 font-mono">
              {hardware?.accelerator_name || 'NVIDIA RTX (CUDA Compute 8.9+)'}
            </span>
          </span>
          <span className="text-slate-600">|</span>
          <span className="flex items-center gap-1.5">
            <HardDrive className="w-3.5 h-3.5 text-indigo-400" />
            <span>Bus:</span>
            <span className="font-mono text-indigo-300">
              {hardware?.is_unified_memory
                ? `Apple Silicon UMA (${hardware.memory_bandwidth_gbps.toFixed(0)} GB/s Zero-Copy)`
                : `PCIe 3.0 x8 (~7.9 GB/s theoretical) | VRAM Peak: ${hardware?.memory_bandwidth_gbps.toFixed(0) || '504'} GB/s`}
            </span>
          </span>
        </div>

        <div className="flex items-center gap-3 font-mono text-[10px]">
          <span className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 text-slate-300">
            Bandwidth Law: v &le; Bandwidth / Model_Size
          </span>
          <span className="px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-500/30 text-emerald-400">
            Strategy: {hardware?.recommended_70b_strategy || 'LayerStream Ping-Pong DMA'}
          </span>
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-7 gap-3 p-4">
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
            <span className="text-2xl font-bold font-mono text-emerald-400">
              {tokPerSec > 0 ? tokPerSec.toFixed(1) : '--'}
            </span>
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
          <span className="text-2xl font-bold font-mono text-cyan-300">
            {ttftMs > 0 ? ttftMs.toFixed(1) : '--'}
          </span>
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
  </div>
  );
};

