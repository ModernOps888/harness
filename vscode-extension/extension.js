'use strict';
const vscode = require('vscode');
const { toOpenAIMessages, toOpenAITools, readSSE, StreamAccumulator, normalizeSchema } = require('./lib/openai');

const VENDOR = 'harness';
/** Rough chars-per-token for code; refined per model from the backend's real prompt_tokens. */
const DEFAULT_CHARS_PER_TOKEN = 3.2;

const num = (v, dflt, lo, hi) => (Number.isFinite(Number(v)) ? Math.min(hi, Math.max(lo, Number(v))) : dflt);

function cfg() {
  const c = vscode.workspace.getConfiguration('harness');
  return {
    endpoint: String(c.get('endpoint', 'http://127.0.0.1:8090/v1')).replace(/\/+$/, ''),
    apiKey: String(c.get('apiKey', '')),
    maxOutputTokens: Math.round(num(c.get('maxOutputTokens', 4096), 4096, 64, 65536)),
    temperature: num(c.get('temperature', 0.2), 0.2, 0, 2),
    topP: num(c.get('topP', 0.8), 0.8, 0, 1),
    topK: Math.round(num(c.get('topK', 20), 20, 0, 200)),
    repeatPenalty: num(c.get('repeatPenalty', 1.05), 1.05, 0.5, 2),
  };
}

function headers(apiKey) {
  const h = { 'Content-Type': 'application/json' };
  if (apiKey) h.Authorization = `Bearer ${apiKey}`;
  return h;
}

/** Abort fetch when VS Code cancels the request. */
function abortOn(token) {
  const ac = new AbortController();
  const sub = token.onCancellationRequested(() => ac.abort());
  return { signal: ac.signal, dispose: () => sub.dispose() };
}

async function errorFromResponse(res) {
  let detail = '';
  try {
    const j = await res.json();
    detail = (j && j.error && j.error.message) || JSON.stringify(j);
  } catch {
    detail = res.statusText;
  }
  return new Error(`HARNESS backend error (HTTP ${res.status}): ${detail}`);
}

class HarnessProvider {
  /** @param {vscode.OutputChannel} log */
  constructor(log) {
    this.log = log;
    this._changed = new vscode.EventEmitter();
    this.onDidChangeLanguageModelChatInformation = this._changed.event;
    /** @type {Map<string, number>} model id -> observed chars per token */
    this.charsPerToken = new Map();
  }

  refresh() {
    this._changed.fire();
  }

  async provideLanguageModelChatInformation(options, token) {
    const { endpoint, apiKey, maxOutputTokens } = cfg();
    const root = endpoint.replace(/\/v1$/, '');
    const { signal, dispose } = abortOn(token);
    try {
      const res = await fetch(`${endpoint}/models`, { headers: headers(apiKey), signal });
      if (!res.ok) throw await errorFromResponse(res);
      const list = await res.json();

      // Real runtime context (n_ctx the engine was launched with), not the training context.
      let nCtx = 0;
      try {
        const pr = await fetch(`${root}/props`, { headers: headers(apiKey), signal });
        if (pr.ok) {
          const props = await pr.json();
          nCtx = Number(props?.default_generation_settings?.n_ctx) || 0;
        }
      } catch {
        /* props is optional */
      }
      const ctx = nCtx || 8192;

      return (list.data || []).map((m) => {
        const id = String(m.id);
        return {
          id,
          name: `${id} (HARNESS local)`,
          family: 'harness',
          version: '1.0.0',
          tooltip: `Local model served by HARNESS. Runtime context ${ctx} tokens (server compacts large MCP tool sets to fit).`,
          detail: `local \u00b7 ${ctx.toLocaleString()} ctx`,
          // Advertise >=128k budget to VS Code's AgentHost / copilot-runtime so its pre-send
          // `compaction_static_context_blocked` check does not abort when 5+ MCP servers are enabled;
          // harness-server compacts incoming tool schemas so the real prompt fits inside `ctx`.
          maxInputTokens: Math.max(128000, ctx - maxOutputTokens),
          maxOutputTokens,
          capabilities: { toolCalling: true, imageInput: false },
        };
      });
    } catch (e) {
      this.log.appendLine(`[models] ${e && e.message ? e.message : e}`);
      if (!options.silent) {
        vscode.window.showWarningMessage(
          `HARNESS engine not reachable at ${endpoint}. Run "HARNESS: Start Engine" or ./launch_qwen32b_coder.ps1 + harness-cli serve.`
        );
      }
      return [];
    } finally {
      dispose();
    }
  }

  async provideLanguageModelChatResponse(model, messages, options, progress, token) {
    const { endpoint, apiKey, temperature, topP, topK, repeatPenalty } = cfg();
    const body = {
      model: model.id,
      messages: toOpenAIMessages(messages),
      stream: true,
      stream_options: { include_usage: true },
      temperature: options.modelOptions?.temperature ?? temperature,
      top_p: topP,
      top_k: topK,
      repeat_penalty: repeatPenalty,
      max_tokens: Math.min(options.modelOptions?.max_tokens ?? model.maxOutputTokens, model.maxOutputTokens),
    };
    const tools = toOpenAITools(options.tools);
    if (tools) {
      body.tools = tools;
      body.tool_choice = options.toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto';
      body.parallel_tool_calls = true;
    }

    const started = Date.now();
    const { signal, dispose } = abortOn(token);
    try {
      const res = await fetch(`${endpoint}/chat/completions`, {
        method: 'POST',
        headers: headers(apiKey),
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok || !res.body) throw await errorFromResponse(res);

      const acc = new StreamAccumulator();
      const schemas = new Map((options.tools || []).map((t) => [t.name, normalizeSchema(t.inputSchema)]));
      let firstTokenAt = 0;
      let sawDone = false;
      await readSSE(
        res.body,
        (data) => {
          if (data === '[DONE]') {
            sawDone = true;
            return;
          }
          let chunk;
          try {
            chunk = JSON.parse(data);
          } catch {
            return;
          }
          if (chunk.error) throw new Error(`HARNESS backend stream error: ${chunk.error.message || JSON.stringify(chunk.error)}`);
          for (const p of acc.push(chunk)) {
            if (!firstTokenAt) firstTokenAt = Date.now();
            progress.report(new vscode.LanguageModelTextPart(p.value));
          }
        },
        signal
      );
      if (signal.aborted) return;
      if (!sawDone && !acc.finishReason) {
        throw new Error('HARNESS backend stream ended unexpectedly (no finish_reason): the response may be truncated.');
      }
      for (const t of acc.flush(schemas)) {
        progress.report(new vscode.LanguageModelToolCallPart(t.callId, t.name, t.input));
      }

      if (acc.finishReason === 'length') {
        progress.report(new vscode.LanguageModelTextPart('\n\n[HARNESS: output truncated at the max output token limit]'));
      }
      const u = acc.usage || {};
      const promptChars = JSON.stringify(body.messages).length + (tools ? JSON.stringify(tools).length : 0);
      if (u.prompt_tokens > 0) this.charsPerToken.set(model.id, promptChars / u.prompt_tokens);
      const tps = acc.timings?.predicted_per_second ?? u.tokens_per_second;
      this.log.appendLine(
        `[chat] ${model.id} prompt=${u.prompt_tokens ?? '?'} completion=${u.completion_tokens ?? '?'} ` +
          `tool_calls=${tools ? (acc.finishReason === 'tool_calls' ? 'yes' : 'no') : 'n/a'} ` +
          `cached=${acc.timings?.cache_n ?? u.prompt_tokens_details?.cached_tokens ?? '?'} ` +
          `ttft=${firstTokenAt ? firstTokenAt - started : '?'}ms total=${Date.now() - started}ms ` +
          `decode=${tps !== undefined ? Number(tps).toFixed(2) : '?'} tok/s`
      );
    } finally {
      dispose();
    }
  }

  async provideTokenCount(model, text, _token) {
    const cpt = this.charsPerToken.get(model.id) || DEFAULT_CHARS_PER_TOKEN;
    let chars = 0;
    if (typeof text === 'string') {
      chars = text.length;
    } else {
      chars = JSON.stringify(toOpenAIMessages([text])).length;
    }
    return Math.ceil(chars / cpt);
  }
}

function activate(context) {
  const log = vscode.window.createOutputChannel('HARNESS');
  const provider = new HarnessProvider(log);
  context.subscriptions.push(
    log,
    vscode.lm.registerLanguageModelChatProvider(VENDOR, provider),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('harness')) provider.refresh();
    }),
    vscode.commands.registerCommand('harness.refreshModels', () => provider.refresh()),
    vscode.commands.registerCommand('harness.showLog', () => log.show(true)),
    vscode.commands.registerCommand('harness.startEngine', () => {
      const script = String(vscode.workspace.getConfiguration('harness').get('launchScript', ''));
      if (!script || !/\.ps1$/i.test(script)) {
        vscode.window.showWarningMessage('Set the machine-level setting "harness.launchScript" to the absolute path of launch_qwen32b_coder.ps1.');
        return;
      }
      const term = vscode.window.createTerminal({
        name: 'HARNESS engine',
        shellPath: 'powershell.exe',
        shellArgs: ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script, '-Mode', 'Agent'],
      });
      term.show();
    })
  );
  return { provider };
}

function deactivate() {}

module.exports = { activate, deactivate };
