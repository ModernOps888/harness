import React, { useEffect, useState } from 'react';
import { Waves, HardDrive, Cpu } from 'lucide-react';

interface Props {
  activeModel: string;
  isStreaming: boolean;
}

export const LayerStreamVisualizer: React.FC<Props> = ({ activeModel, isStreaming }) => {
  const [currentLayer, setCurrentLayer] = useState(15);
  const lower = activeModel.toLowerCase();
  const is70b = lower.includes('70b');
  const is30bMoe = lower.includes('30b') || lower.includes('moe');
  const is14b = lower.includes('14b');

  const totalLayers = is70b ? 80 : is30bMoe || is14b ? 48 : 64;
  const gpuOffloadedLayers = is70b ? 11 : is30bMoe ? 48 : is14b ? 38 : 20;

  useEffect(() => {
    if (!isStreaming) return;
    const interval = setInterval(() => {
      setCurrentLayer((prev) => (prev + 1) % totalLayers);
    }, 120);
    return () => clearInterval(interval);
  }, [isStreaming, totalLayers]);

  return (
    <div className="bg-[#0b0f19] border border-slate-800 rounded-xl p-4 my-2">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Waves className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            {is30bMoe
              ? 'Qwen3-Coder-30B-A3B Hybrid Sparse MoE Schedule (48/48 Attn + 16/48 Full MoE in 8GB GDDR7)'
              : is70b
              ? '70B Layer Streaming Schedule (80 Layers, 11/81 GPU + 1B Drafter in VRAM)'
              : is14b
              ? 'Qwen2.5-Coder-14B Speculative Schedule (38/49 Layers + 100% 0.5B Drafter in 8GB GDDR7)'
              : 'Qwen2.5-Coder-32B Ultra-Parallel Speculative Schedule (20/65 GPU Layers + 100% 0.5B Drafter in GDDR7)'}
          </h3>
        </div>
        <span className="text-xs font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2 py-0.5 rounded">
          {is30bMoe
            ? '-ngl 99 --n-cpu-moe 32 | 16.37 tok/s pooled / 16.53 tok/s peak'
            : is70b
            ? 'Ping-Pong Double Buffer DMA (1.68 tok/s)'
            : is14b
            ? '-ngl 38 + 0.5B Drafter | 15.89 tok/s pooled / 19.17 tok/s peak'
            : '-ngl 20 + 0.5B Drafter | 3.90 tok/s pooled / 4.80 tok/s peak'}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
        {/* GPU Slot 0 (Compute) */}
        <div className="bg-[#101726] border border-emerald-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <Cpu className="w-3.5 h-3.5" />
              {is30bMoe
                ? 'Slot 0: RTX 5060 8GB GDDR7 (100% Attn/KV/Router + 16 Full MoE Layers, 6,463 MiB)'
                : `Slot 0: RTX 5060 8GB Active GDDR7 (${gpuOffloadedLayers}/${totalLayers + 1} Layers)`}
            </span>
            <span className="text-[11px] font-mono text-slate-400">Layer #{currentLayer}</span>
          </div>
          <p className="text-xs text-slate-400">
            {is30bMoe
              ? 'Executing 48-layer GQA FlashAttention (Q8_0 K / Q4_0 V) + Top-8 Softmax Router + 16 Resident MoE Layers at 504 GB/s'
              : 'Executing GQA 40:8 FlashAttention (Q8_0 K / Q4_0 V) + 100% GPU-Resident 0.5B Speculative Drafter (--no-op-offload)'}
          </p>
        </div>

        {/* GPU Slot 1 (Prefetch / Active Experts) */}
        <div className="bg-[#101726] border border-cyan-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
              <HardDrive className="w-3.5 h-3.5" />
              {is30bMoe
                ? 'Slot 1: Dual-Channel DDR4 Active Expert Stream (32 MoE Layers, 8/128 Active)'
                : 'Slot 1: Dual-Channel DDR4 Batched M=K+1 Speculative Verification'}
            </span>
            <span className="text-[11px] font-mono text-slate-400">Layer #{(currentLayer + 1) % totalLayers}</span>
          </div>
          <p className="text-xs text-slate-400">
            {is30bMoe
              ? 'Reading only 6.25% (8/128) routed experts per token (~0.70 GB/tok vs 14.46 GB Dense 32B -> 95.2% DDR4 traffic eliminated)'
              : 'Batch-verifying K=7 draft tokens in 1 weight sweep (85.4% pooled / 96.2% Rust draft acceptance, 40 graph splits)'}
          </p>
        </div>
      </div>

      {/* Layer Progress Ribbon */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-[11px] text-slate-400 font-mono">
          <span>Layer Execution Sequence ({activeModel})</span>
          <span>
            {currentLayer + 1} / {totalLayers} Layers Completed
          </span>
        </div>
        <div className="grid grid-cols-32 gap-0.5 h-2 w-full bg-slate-900 rounded overflow-hidden p-0.5">
          {Array.from({ length: 32 }).map((_, idx) => {
            const scale = totalLayers / 32;
            const mappedIdx = Math.floor(idx * scale);
            const isDone = mappedIdx < currentLayer;
            const isCurrent = Math.abs(mappedIdx - currentLayer) < scale;
            return (
              <div
                key={idx}
                className={`h-full rounded-sm transition-all duration-100 ${
                  isCurrent
                    ? 'bg-cyan-400 shadow-sm shadow-cyan-400'
                    : isDone
                    ? 'bg-emerald-600/70'
                    : 'bg-slate-800'
                }`}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
};
