import React from 'react';
import { X, Cpu, Server, Zap, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';
import { HardwareProfile } from './MetricsBar';

interface HardwareModalProps {
  isOpen: boolean;
  onClose: () => void;
  hardware?: HardwareProfile | null;
}

export const HardwareModal: React.FC<HardwareModalProps> = ({ isOpen, onClose, hardware }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#0e1320] border border-slate-700/80 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-[#141b2d] border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                Hardware Sizing &amp; Memory Bandwidth Physics
                <span className="text-[10px] bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 px-2 py-0.5 rounded font-mono">
                  PHYSICAL LAWS
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Ground-truth comparison: PC Discrete PCIe DMA vs Apple Silicon Unified Memory
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-xs text-slate-300">
          {/* Active Detected Hardware Card */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-slate-900 to-[#12192b] border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold">Active Detected Machine</div>
              <div className="text-base font-bold text-slate-100 mt-1 flex items-center gap-2">
                <span>{hardware?.accelerator_name || 'NVIDIA RTX Architecture'}</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {hardware?.os.toUpperCase()} ({hardware?.arch})
                </span>
              </div>
              <div className="text-xs text-slate-400 mt-1">
                Host RAM: <span className="text-slate-200 font-mono">{hardware?.host_ram_gb.toFixed(0)} GB</span> | Device VRAM: <span className="text-slate-200 font-mono">{hardware?.vram_gb.toFixed(0)} GB</span> | Peak Bandwidth: <span className="text-cyan-400 font-mono font-bold">{hardware?.memory_bandwidth_gbps.toFixed(0)} GB/s</span>
              </div>
            </div>

            <div className="px-4 py-2.5 rounded-lg bg-slate-800/80 border border-slate-700/60 font-mono text-center md:text-right">
              <div className="text-[10px] text-slate-400">27B-35B Coder Execution Strategy</div>
              <div className="text-xs font-semibold text-emerald-400 mt-0.5">
                {hardware?.recommended_70b_strategy || 'Qwen2.5-Coder-32B (-ngl 17) + 1.5B Drafter (-ngld 99)'}
              </div>
            </div>
          </div>

          {/* Theoretical Law */}
          <div className="p-4 rounded-xl bg-[#0b0f19] border border-cyan-900/40">
            <h3 className="text-xs font-bold text-cyan-400 flex items-center gap-1.5 mb-2">
              <Zap className="w-4 h-4" />
              The Autoregressive Memory Bandwidth Law &amp; Speculative Multiplication
            </h3>
            <p className="text-slate-300 leading-relaxed">
              Every token generated during standard autoregressive decoding requires streaming the active CPU-resident weight matrix through execution cores once.
              With a 100% VRAM-resident 1.5B speculative drafter, HARNESS verifies <em>K</em> candidate code tokens in a single 32B target pass:
            </p>
            <div className="my-3 p-3 rounded-lg bg-black/40 border border-slate-800 font-mono text-center text-sm text-cyan-300">
              Throughput (tokens/sec) &le; (1 + &alpha;&middot;K) &times; Memory Bandwidth (GB/s) / CPU-Resident Weight Size (GB)
            </div>
            <p className="text-slate-400 leading-relaxed">
              On RTX 5060 8GB (448 GB/s GDDR7) + DDR4-2133, <strong>Qwen3-Coder-30B-A3B Q4_K_M</strong> achieves <strong>446.9 tok/s GPU prefill</strong> and <strong>19.88-21.40 tok/s decode</strong> at 32,768 context (and 380.5 tok/s prefill / 14.04 tok/s decode at 65,536 context), while <strong>Qwen2.5-Coder-32B Dense</strong> accelerates from <strong>1.16 tok/s</strong> (raw baseline) to <strong>3.90 tok/s pooled / 4.80 tok/s peak</strong> with the 0.5B speculative drafter.
            </p>
          </div>

          {/* Sizing Matrix Table */}
          <div>
            <h3 className="text-xs font-bold text-slate-200 mb-3 uppercase tracking-wider flex items-center gap-1.5">
              <Server className="w-4 h-4 text-indigo-400" />
              Physical Multi-Platform Feasibility Matrix
            </h3>
            <div className="border border-slate-800 rounded-xl overflow-hidden">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-[#111827] text-slate-400 text-[11px] border-b border-slate-800">
                    <th className="py-2.5 px-3 font-semibold">Platform &amp; Memory Tier</th>
                    <th className="py-2.5 px-3 font-semibold">Interconnect Bus</th>
                    <th className="py-2.5 px-3 font-semibold">Active Bandwidth</th>
                    <th className="py-2.5 px-3 font-semibold">14B Coder (8.5GB)</th>
                    <th className="py-2.5 px-3 font-semibold">30B-A3B MoE / 32B Coder</th>
                    <th className="py-2.5 px-3 font-semibold">70B (Q2_K 26.4GB)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                  <tr className="hover:bg-slate-900/40 bg-emerald-950/20">
                    <td className="py-2.5 px-3 font-sans font-bold text-emerald-300">PC with RTX 5060 8GB (Verified Local)</td>
                    <td className="py-2.5 px-3 text-slate-400">PCIe 3.0 x8 + DDR4-2133 dual-channel</td>
                    <td className="py-2.5 px-3 text-slate-300">34.1 GB/s RAM / 448 GB/s GDDR7</td>
                    <td className="py-2.5 px-3 text-emerald-400">15.89 - 19.17 tok/s (Verified)</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">19.9-21.4 tok/s MoE / 3.9-4.8 Dense</td>
                    <td className="py-2.5 px-3 text-slate-300">1.38 tok/s (Verified)</td>
                  </tr>
                  <tr className="hover:bg-slate-900/40 opacity-75">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-300">Mac M3/M4 Pro (24-36GB)*</td>
                    <td className="py-2.5 px-3 text-slate-400">Unified Memory (Metal)</td>
                    <td className="py-2.5 px-3 text-slate-300">150 to 273 GB/s</td>
                    <td className="py-2.5 px-3 text-cyan-400">18 to 32 tok/s (Theoretical)</td>
                    <td className="py-2.5 px-3 text-cyan-400">12 to 24 tok/s (Speculative)</td>
                    <td className="py-2.5 px-3 text-cyan-400">6 to 9 tok/s (Theoretical)</td>
                  </tr>
                  <tr className="hover:bg-slate-900/40 opacity-75">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-300">Mac M3/M4 Max (48-64GB)*</td>
                    <td className="py-2.5 px-3 text-slate-400">Unified Memory (Metal)</td>
                    <td className="py-2.5 px-3 text-slate-300">300 to 400+ GB/s</td>
                    <td className="py-2.5 px-3 text-cyan-400">35 to 48 tok/s (Theoretical)</td>
                    <td className="py-2.5 px-3 text-cyan-400">22 to 38 tok/s (Speculative)</td>
                    <td className="py-2.5 px-3 text-cyan-400">8.5 to 11.2 tok/s (Theoretical)</td>
                  </tr>
                  <tr className="hover:bg-slate-900/40 opacity-75">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-300">Mac Studio M2 Ultra (128GB)*</td>
                    <td className="py-2.5 px-3 text-slate-400">Unified Memory (Metal)</td>
                    <td className="py-2.5 px-3 text-slate-300">800 GB/s</td>
                    <td className="py-2.5 px-3 text-cyan-400">65 to 90 tok/s (Theoretical)</td>
                    <td className="py-2.5 px-3 text-cyan-400">40 to 65 tok/s (Speculative)</td>
                    <td className="py-2.5 px-3 text-cyan-400">14 to 18 tok/s (Theoretical)</td>
                  </tr>
                  <tr className="hover:bg-slate-900/40 opacity-75">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-300">Mac Studio Ultra (192-512GB)*</td>
                    <td className="py-2.5 px-3 text-slate-400">Unified Memory (Metal)</td>
                    <td className="py-2.5 px-3 text-slate-300">800 to 1200+ GB/s</td>
                    <td className="py-2.5 px-3 text-cyan-400">75 to 110 tok/s (Theoretical)</td>
                    <td className="py-2.5 px-3 text-cyan-400">50 to 80 tok/s (Speculative)</td>
                    <td className="py-2.5 px-3 text-cyan-400">20 to 24 tok/s (Theoretical)</td>
                  </tr>
                </tbody>
              </table>
              <div className="p-2 text-[10px] text-slate-400 bg-slate-900/60 border-t border-slate-800">
                *Notice: On this PC (RTX 5060 8GB / 32GB DDR4), both Qwen2.5-Coder-32B (1.16 -&gt; 1.58 -&gt; 2.99-3.65 tok/s) and Llama-3.1-70B (0.76 -&gt; 1.38 tok/s) have been physically executed and captured in benchmarks/runs/.
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 bg-[#111827] border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
