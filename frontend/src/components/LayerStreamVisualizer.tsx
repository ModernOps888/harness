import React, { useEffect, useState } from 'react';
import { Waves, ArrowRight, HardDrive, Cpu, CheckCircle2 } from 'lucide-react';

interface Props {
  activeModel: string;
  isStreaming: boolean;
}

export const LayerStreamVisualizer: React.FC<Props> = ({ activeModel, isStreaming }) => {
  const [currentLayer, setCurrentLayer] = useState(0);
  const is70b = activeModel.toLowerCase().includes('70b');
  const is14bOr30b = activeModel.toLowerCase().includes('14b') || activeModel.toLowerCase().includes('30b');
  const totalLayers = is70b ? 80 : is14bOr30b ? 48 : 64;
  const gpuOffloadedLayers = is70b ? 19 : is14bOr30b ? 38 : 17;

  useEffect(() => {
    if (!isStreaming) return;
    const interval = setInterval(() => {
      setCurrentLayer((prev) => (prev + 1) % totalLayers);
    }, 120);
    return () => clearInterval(interval);
  }, [isStreaming, totalLayers]);

  return (
    <div className="bg-[#0b0f19] border border-slate-800 rounded-xl p-4 my-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Waves className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            {is70b
              ? '70B Layer Streaming Schedule (80 Layers, 19/81 GPU Offloaded)'
              : '27B-35B Coder Hybrid Speculative Schedule (64 Layers, 17/65 GPU + 1.5B Drafter 100% VRAM)'}
          </h3>
        </div>
        <span className="text-xs font-mono text-cyan-400 bg-cyan-950/40 border border-cyan-800/40 px-2 py-0.5 rounded">
          {is70b ? 'Ping-Pong Double Buffer DMA' : 'Q8_0/Q4_0 FlashAttn + 1.5B Speculative Draft'}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
        {/* GPU Slot 0 (Compute) */}
        <div className="bg-[#101726] border border-emerald-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <Cpu className="w-3.5 h-3.5" />
              Slot 0: RTX 5060 8GB Active VRAM ({gpuOffloadedLayers}/{totalLayers + 1} Layers)
            </span>
            <span className="text-[11px] font-mono text-slate-400">Layer #{currentLayer}</span>
          </div>
          <p className="text-xs text-slate-400">
            {is70b
              ? 'Executing GQA 64:8 FlashAttention + SwiGLU in 6.5 GB VRAM footprint'
              : 'Executing GQA 40:8 FlashAttention (Q8_0 K / Q4_0 V) + 1.5B Drafter (29/29 layers in VRAM)'}
          </p>
        </div>

        {/* GPU Slot 1 (Prefetch) */}
        <div className="bg-[#101726] border border-cyan-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
              <HardDrive className="w-3.5 h-3.5" />
              Slot 1: Dual-Channel DDR4 + Speculative Verification
            </span>
            <span className="text-[11px] font-mono text-slate-400">Layer #{(currentLayer + 1) % totalLayers}</span>
          </div>
          <p className="text-xs text-slate-400">
            {is70b
              ? 'Zero-copy PCIe stream transfer from system RAM directly to VRAM buffer'
              : 'Batch-verifying K=8 draft tokens in 1 pass (68.7% draft acceptance -> 2.99-3.65 tok/s)'}
          </p>
        </div>
      </div>

      {/* Layer Progress Ribbon */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-[11px] text-slate-400 font-mono">
          <span>Layer Execution Sequence ({activeModel})</span>
          <span>{currentLayer + 1} / {totalLayers} Layers Completed</span>
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
