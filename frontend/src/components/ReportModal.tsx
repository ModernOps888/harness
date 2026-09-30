import React, { useEffect, useState } from 'react';
import { X, Download, Copy, Check, FileText, Cpu, ShieldCheck } from 'lucide-react';

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ReportModal: React.FC<ReportModalProps> = ({ isOpen, onClose }) => {
  const [reportMarkdown, setReportMarkdown] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    if (!isOpen) return;

    const fetchReport = async () => {
      setLoading(true);
      try {
        const res = await fetch('/report');
        if (res.ok) {
          const text = await res.text();
          setReportMarkdown(text);
        } else {
          setReportMarkdown('# Benchmark Report\n\nFailed to fetch live report from `/report` endpoint.');
        }
      } catch (err) {
        setReportMarkdown('# Benchmark Report\n\nError connecting to HARNESS server daemon.');
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [isOpen]);

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(reportMarkdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([reportMarkdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'HARNESS_BENCHMARK_REPORT.md';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#0e1320] border border-slate-700/80 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-[#141b2d] border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                HARNESS Scientific Benchmark &amp; Telemetry Audit
                <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded font-mono">
                  VERIFIED
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Ground-truth physical hardware metrics, memory bandwidth bounds, and bio-module telemetry
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors border border-slate-700"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
              <span>{copied ? 'Copied' : 'Copy Markdown'}</span>
            </button>

            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium transition-colors shadow-lg shadow-cyan-600/20"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download (.md)</span>
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors ml-2"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 font-mono text-xs text-slate-300 bg-[#0a0e17] leading-relaxed selection:bg-cyan-500/30">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-20 gap-3 text-slate-500">
              <Cpu className="w-8 h-8 animate-spin text-cyan-400" />
              <span>Auditing live hardware telemetry &amp; generating report...</span>
            </div>
          ) : (
            <pre className="whitespace-pre-wrap font-sans text-xs bg-slate-900/60 p-5 rounded-xl border border-slate-800/80 text-slate-200 leading-relaxed overflow-x-auto">
              {reportMarkdown}
            </pre>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 bg-[#111827] border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
          <span className="flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            All metrics computed from first-principles memory bandwidth and live Rust benchmarks
          </span>
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
