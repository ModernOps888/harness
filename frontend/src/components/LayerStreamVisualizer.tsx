import React, { useEffect, useState } from 'react';
import { Waves, ArrowRight, HardDrive, Cpu, CheckCircle2 } from 'lucide-react';

interface Props {
  activeModel: string;
  isStreaming: boolean;
}

export const LayerStreamVisualizer: React.FC<Props> = ({ activeModel, isStreaming }) => {
  const [currentLayer, setCurrentLayer] = useState(0);
  const totalLayers = 80;

  useEffect(() => {
    if (!isStreaming) return;
    const interval = setInterval(() => {
      setCurrentLayer((prev) => (prev + 1) % totalLayers);
    }, 120);
    return () => clearInterval(interval);
  }, [isStreaming]);

  const is70b = activeModel.includes('70B');

  return (
    <div className="bg-[#0b0f19] border border-slate-800 rounded-xl p-4 my-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Waves className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Temporal Layer Streaming Architecture (70B on 8GB VRAM)
          </h3>
        </div>
        <span className="text-xs font-mono text-cyan-400 bg-cyan-950/40 border border-cyan-800/40 px-2 py-0.5 rounded">
          Ping-Pong Double Buffer DMA
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
        {/* GPU Slot 0 (Compute) */}
        <div className="bg-[#101726] border border-emerald-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <Cpu className="w-3.5 h-3.5" />
              Slot 0: Active Compute
            </span>
            <span className="text-[11px] font-mono text-slate-400">Layer #{currentLayer}</span>
          </div>
          <p className="text-xs text-slate-400">
            Executing FlashAttention v3 + SwiGLU forward pass in 4.8GB VRAM footprint
          </p>
        </div>

        {/* GPU Slot 1 (Prefetch) */}
        <div className="bg-[#101726] border border-cyan-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
              <HardDrive className="w-3.5 h-3.5" />
              Slot 1: Async DMA Prefetch
            </span>
            <span className="text-[11px] font-mono text-slate-400">Layer #{(currentLayer + 1) % totalLayers}</span>
          </div>
          <p className="text-xs text-slate-400">
            Zero-copy PCIe stream transfer from system RAM directly to VRAM buffer
          </p>
        </div>
      </div>

      {/* Layer Progress Ribbon */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-[11px] text-slate-400 font-mono">
          <span>Layer Streaming Sequence</span>
          <span>{currentLayer + 1} / {totalLayers} Layers Completed</span>
        </div>
        <div className="grid grid-cols-40 gap-0.5 h-2 w-full bg-slate-900 rounded overflow-hidden p-0.5">
          {Array.from({ length: 40 }).map((_, idx) => {
            const mappedIdx = idx * 2;
            const isDone = mappedIdx < currentLayer;
            const isCurrent = mappedIdx === currentLayer || mappedIdx + 1 === currentLayer;
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
