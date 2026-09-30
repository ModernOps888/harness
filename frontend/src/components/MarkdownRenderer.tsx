import React, { useState } from 'react';
import { Copy, Check, Terminal, Code2, Sigma } from 'lucide-react';

interface Props {
  content: string;
}

export const MarkdownRenderer: React.FC<Props> = ({ content }) => {
  // Split into code blocks vs text blocks
  const parts = parseMarkdown(content);

  return (
    <div className="space-y-3 font-sans leading-relaxed text-slate-200">
      {parts.map((part, index) => {
        if (part.type === 'code') {
          return (
            <CodeBlock
              key={index}
              language={part.language || 'text'}
              code={part.content}
            />
          );
        } else {
          return <RichText key={index} text={part.content} />;
        }
      })}
    </div>
  );
};

interface Block {
  type: 'text' | 'code';
  content: string;
  language?: string;
}

function parseMarkdown(text: string): Block[] {
  const blocks: Block[] = [];
  const regex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({
        type: 'text',
        content: text.slice(lastIndex, match.index),
      });
    }
    blocks.push({
      type: 'code',
      language: match[1].toLowerCase() || 'code',
      content: match[2].trimEnd(),
    });
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    blocks.push({
      type: 'text',
      content: text.slice(lastIndex),
    });
  }

  return blocks;
}

const CodeBlock: React.FC<{ language: string; code: string }> = ({ language, code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines = code.split('\n');

  return (
    <div className="my-3 rounded-xl border border-slate-800/90 bg-[#090d16] overflow-hidden shadow-xl shadow-black/40">
      {/* Code Header Bar */}
      <div className="flex items-center justify-between px-3.5 py-2 bg-[#0d1322] border-b border-slate-800/80 text-xs">
        <div className="flex items-center gap-2">
          {language === 'bash' || language === 'sh' || language === 'shell' ? (
            <Terminal className="w-3.5 h-3.5 text-amber-400" />
          ) : (
            <Code2 className="w-3.5 h-3.5 text-emerald-400" />
          )}
          <span className="font-mono font-semibold uppercase tracking-wider text-slate-300">
            {language}
          </span>
          <span className="text-[10px] text-slate-500 font-mono">
            {lines.length} lines
          </span>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/40 text-slate-300 hover:text-white transition-all text-[11px] font-mono"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3 text-slate-400" />
              <span>Copy Code</span>
            </>
          )}
        </button>
      </div>

      {/* Code Body with Line Numbers */}
      <div className="p-3.5 overflow-x-auto text-[13px] font-mono leading-relaxed">
        <div className="table w-full">
          {lines.map((line, idx) => (
            <div key={idx} className="table-row hover:bg-slate-800/20 group">
              <span className="table-cell select-none pr-4 text-right text-slate-600 group-hover:text-slate-500 text-[11px] w-8">
                {idx + 1}
              </span>
              <span className="table-cell whitespace-pre text-slate-200">
                {highlightSyntax(line, language)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Lightweight syntax token colorizer for code blocks
function highlightSyntax(line: string, _language: string): React.ReactNode {
  // Comments
  if (line.trim().startsWith('//') || line.trim().startsWith('#') || line.trim().startsWith('/*')) {
    return <span className="text-slate-500 italic">{line}</span>;
  }

  // Keywords
  const keywords = /\b(fn|pub|struct|impl|enum|let|mut|if|else|match|for|in|while|return|use|mod|const|static|async|await|type|class|interface|def|import|from|export|function|const|val|var)\b/g;
  const types = /\b(Self|usize|u8|u16|u32|u64|i8|i16|i32|i64|f32|f64|bool|str|String|Vec|Option|Result|Some|None|Ok|Err|Tensor|Device|Arc|Mutex|RwLock)\b/g;

  // Render tokens
  const parts = line.split(/("[^"]*"|'[^']*'|\b(?:fn|pub|struct|impl|enum|let|mut|if|else|match|for|in|while|return|use|mod|const|static|async|await|type|class|interface|def|import|from|export|function|const|val|var|Self|usize|u8|u16|u32|u64|i8|i16|i32|i64|f32|f64|bool|str|String|Vec|Option|Result|Some|None|Ok|Err|Tensor|Device|Arc|Mutex|RwLock)\b)/g);

  return parts.map((part, i) => {
    if (part.startsWith('"') || part.startsWith("'")) {
      return <span key={i} className="text-emerald-300">{part}</span>;
    }
    if (keywords.test(part)) {
      return <span key={i} className="text-purple-400 font-semibold">{part}</span>;
    }
    if (types.test(part)) {
      return <span key={i} className="text-cyan-400">{part}</span>;
    }
    return <span key={i}>{part}</span>;
  });
}

const RichText: React.FC<{ text: string }> = ({ text }) => {
  const lines = text.split('\n');

  return (
    <div className="space-y-2">
      {lines.map((line, idx) => {
        const trimmed = line.trim();

        // Empty line
        if (!trimmed) {
          return <div key={idx} className="h-1.5" />;
        }

        // Horizontal Rule
        if (trimmed === '---' || trimmed === '***') {
          return <hr key={idx} className="border-slate-800 my-3" />;
        }

        // Header 3
        if (trimmed.startsWith('### ')) {
          return (
            <h3 key={idx} className="text-base font-bold text-slate-100 mt-4 mb-2 flex items-center gap-2 border-b border-slate-800/80 pb-1">
              <span className="w-1.5 h-4 bg-emerald-500 rounded-full" />
              {renderFormattedSpans(trimmed.slice(4))}
            </h3>
          );
        }

        // Header 4
        if (trimmed.startsWith('#### ')) {
          return (
            <h4 key={idx} className="text-sm font-semibold text-cyan-300 mt-3 mb-1.5 flex items-center gap-2">
              <span className="w-1 h-3 bg-cyan-400 rounded-full" />
              {renderFormattedSpans(trimmed.slice(5))}
            </h4>
          );
        }

        // Math block: $$ ... $$
        if (trimmed.startsWith('$$') && trimmed.endsWith('$$')) {
          const formula = trimmed.slice(2, -2).trim();
          return (
            <div key={idx} className="my-2.5 p-3 rounded-lg bg-[#0c1220] border border-cyan-500/20 text-center font-mono text-cyan-300 flex items-center justify-center gap-2 shadow-inner">
              <Sigma className="w-4 h-4 text-cyan-400 shrink-0" />
              <span className="text-sm tracking-wide">{formula}</span>
            </div>
          );
        }

        // Markdown Table Row (detects tables)
        if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
          const cells = trimmed
            .split('|')
            .slice(1, -1)
            .map((c) => c.trim());
          const isSeparator = cells.every((c) => /^[-:\s]+$/.test(c));

          if (isSeparator) {
            return null; // separator handled visually
          }

          const isHeader = idx > 0 && lines[idx + 1] && lines[idx + 1].includes('---');

          return (
            <div
              key={idx}
              className={`grid gap-2 p-2 text-xs rounded border border-slate-800/60 font-mono ${
                isHeader
                  ? 'bg-slate-800/60 font-bold text-slate-200 border-slate-700'
                  : 'bg-slate-900/30 text-slate-300 hover:bg-slate-800/20'
              }`}
              style={{ gridTemplateColumns: `repeat(${cells.length}, minmax(0, 1fr))` }}
            >
              {cells.map((cell, cIdx) => (
                <div key={cIdx} className="overflow-hidden text-ellipsis">
                  {renderFormattedSpans(cell)}
                </div>
              ))}
            </div>
          );
        }

        // Bullet point
        if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
          return (
            <div key={idx} className="flex items-start gap-2 pl-2 text-sm text-slate-300">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0 mt-2" />
              <div className="flex-1">{renderFormattedSpans(trimmed.slice(2))}</div>
            </div>
          );
        }

        // Numbered item
        const numMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
        if (numMatch) {
          return (
            <div key={idx} className="flex items-start gap-2 pl-2 text-sm text-slate-300">
              <span className="font-mono text-xs text-cyan-400 font-semibold shrink-0 mt-0.5">
                {numMatch[1]}.
              </span>
              <div className="flex-1">{renderFormattedSpans(numMatch[2])}</div>
            </div>
          );
        }

        // Standard paragraph line
        return (
          <p key={idx} className="text-sm text-slate-300 leading-relaxed">
            {renderFormattedSpans(trimmed)}
          </p>
        );
      })}
    </div>
  );
};

// Helper: formats bold (**text**), inline code (`code`), and inline math ($math$)
function renderFormattedSpans(text: string): React.ReactNode {
  // Regex splitting by: **bold**, `inline code`, $math$
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\$[^$]+\$)/g);

  return tokens.map((token, i) => {
    if (token.startsWith('**') && token.endsWith('**')) {
      return (
        <strong key={i} className="font-semibold text-slate-100">
          {token.slice(2, -2)}
        </strong>
      );
    }
    if (token.startsWith('`') && token.endsWith('`')) {
      return (
        <code
          key={i}
          className="px-1.5 py-0.5 mx-0.5 rounded bg-[#151c2e] border border-slate-700/60 font-mono text-[12px] text-emerald-300"
        >
          {token.slice(1, -1)}
        </code>
      );
    }
    if (token.startsWith('$') && token.endsWith('$')) {
      return (
        <span
          key={i}
          className="px-1 py-0.2 mx-0.5 font-mono text-xs text-cyan-300 bg-cyan-950/40 rounded border border-cyan-800/40"
        >
          {token.slice(1, -1)}
        </span>
      );
    }
    return token;
  });
}
