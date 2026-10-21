'use strict';
/**
 * Pure (VS Code-independent) helpers that translate between the VS Code Language Model Chat
 * Provider data model and the OpenAI chat-completions wire format served by HARNESS.
 * Kept free of `require('vscode')` so they are unit-testable with plain `node --test`.
 */

const ROLE_USER = 1;
const ROLE_ASSISTANT = 2;
const ROLE_SYSTEM = 3; // not in the stable API today; handled defensively

/** @param {any} p */
const isToolCallPart = (p) => p && typeof p.callId === 'string' && typeof p.name === 'string' && 'input' in p;
/** @param {any} p */
const isToolResultPart = (p) => p && typeof p.callId === 'string' && Array.isArray(p.content) && !('name' in p);
/** @param {any} p */
const isTextPart = (p) => p && typeof p.value === 'string' && !('callId' in p) && !('mimeType' in p);
/** @param {any} p */
const isDataPart = (p) => p && p.data instanceof Uint8Array && typeof p.mimeType === 'string';

/** Extract plain text from the content array of a tool result. */
function toolResultText(content) {
  const out = [];
  for (const c of content) {
    if (typeof c === 'string') {
      out.push(c);
    } else if (isTextPart(c)) {
      out.push(c.value);
    } else if (isDataPart(c)) {
      if (/^(text\/|application\/json)/.test(c.mimeType)) {
        out.push(Buffer.from(c.data).toString('utf8'));
      } else {
        out.push(`[binary ${c.mimeType}, ${c.data.length} bytes omitted: text-only model]`);
      }
    } else if (c && c.value !== undefined) {
      // e.g. LanguageModelPromptTsxPart: serialise whatever the tool produced
      out.push(typeof c.value === 'string' ? c.value : JSON.stringify(c.value));
    }
  }
  return out.join('\n');
}

/**
 * VS Code messages -> OpenAI messages.
 * Tool results (User-role parts) become `role: "tool"` messages and are emitted *before* any
 * accompanying user text so they directly follow the assistant `tool_calls` turn.
 * @param {ReadonlyArray<{role:number, content: ReadonlyArray<any>}>} messages
 */
function toOpenAIMessages(messages) {
  const out = [];
  for (const m of messages) {
    if (m.role === ROLE_SYSTEM) {
      const sys = m.content.filter(isTextPart).map((p) => p.value).join('\n');
      if (sys.length) out.push({ role: 'system', content: sys });
    } else if (m.role === ROLE_ASSISTANT) {
      const text = m.content.filter(isTextPart).map((p) => p.value).join('');
      const calls = m.content.filter(isToolCallPart).map((p) => ({
        id: p.callId,
        type: 'function',
        function: { name: p.name, arguments: JSON.stringify(p.input ?? {}) },
      }));
      /** @type {any} */
      const msg = { role: 'assistant', content: text.length ? text : calls.length ? null : '' };
      if (calls.length) msg.tool_calls = calls;
      out.push(msg);
    } else {
      for (const p of m.content) {
        if (isToolResultPart(p)) {
          out.push({ role: 'tool', tool_call_id: p.callId, content: toolResultText(p.content) });
        }
      }
      const text = m.content
        .map((p) => {
          if (isTextPart(p)) return p.value;
          if (isDataPart(p) && /^image\//.test(p.mimeType)) return `[image ${p.mimeType} omitted: text-only model]`;
          if (isDataPart(p) && /^(text\/|application\/json)/.test(p.mimeType)) return Buffer.from(p.data).toString('utf8');
          return '';
        })
        .filter((s) => s.length)
        .join('\n');
      if (text.length) out.push({ role: 'user', content: text });
    }
  }
  return out;
}

/**
 * Qwen's chat template iterates `parameters.properties`; a schema without `properties` (or a
 * non-object schema) makes the Jinja render fail for the whole request. Normalise defensively.
 */
function normalizeSchema(schema) {
  const out = schema && typeof schema === 'object' && !Array.isArray(schema) ? { ...schema } : {};
  out.type = 'object';
  if (!out.properties || typeof out.properties !== 'object' || Array.isArray(out.properties)) out.properties = {};
  if (out.required !== undefined && !Array.isArray(out.required)) delete out.required;
  return out;
}

/** @param {ReadonlyArray<{name:string, description:string, inputSchema?:object}>|undefined} tools */
function toOpenAITools(tools) {
  if (!tools || !tools.length) return undefined;
  return tools.map((t) => ({
    type: 'function',
    function: { name: t.name, description: t.description || '', parameters: normalizeSchema(t.inputSchema) },
  }));
}

/**
 * Qwen3-Coder emits `<parameter>` values as raw strings; llama.cpp infers types from the schema but
 * cannot for typeless / anyOf properties. Coerce string values to the declared JSON type so VS Code's
 * tool input validation does not reject e.g. "42" for an integer or "true" for a boolean.
 */
function coerceArgs(input, schema) {
  if (!input || typeof input !== 'object' || !schema || !schema.properties) return input;
  const out = { ...input };
  for (const [k, v] of Object.entries(out)) {
    const prop = schema.properties[k];
    if (!prop || typeof v !== 'string') continue;
    const types = Array.isArray(prop.type) ? prop.type : prop.type ? [prop.type] : [];
    if (types.includes('string')) continue;
    try {
      if ((types.includes('integer') || types.includes('number')) && v.trim() !== '' && !Number.isNaN(Number(v))) out[k] = Number(v);
      else if (types.includes('boolean') && /^(true|false)$/i.test(v.trim())) out[k] = v.trim().toLowerCase() === 'true';
      else if ((types.includes('array') || types.includes('object')) ) out[k] = JSON.parse(v);
    } catch {
      /* leave the raw string; the tool will report a precise validation error */
    }
  }
  return out;
}

/** Parse an SSE byte stream, invoking `onData(jsonString|'[DONE]')` for each `data:` event. */
async function readSSE(body, onData, signal) {
  const decoder = new TextDecoder();
  let buf = '';
  for await (const chunk of body) {
    if (signal && signal.aborted) return;
    buf += decoder.decode(chunk, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).replace(/\r$/, '');
      buf = buf.slice(idx + 1);
      if (!line.startsWith('data:')) continue;
      onData(line.slice(5).trim());
    }
  }
  const tail = buf.trim();
  if (tail.startsWith('data:')) onData(tail.slice(5).trim());
}

let idCounter = 0;
const genCallId = () => `call_${Date.now().toString(36)}${(idCounter++).toString(36)}`;

/**
 * Stateful accumulator for OpenAI streaming chunks. `push(chunk)` returns the response parts that
 * can be emitted immediately (text deltas); `flush()` returns completed tool calls.
 */
class StreamAccumulator {
  constructor() {
    /** @type {Map<number,{id:string,name:string,args:string}>} */
    this.calls = new Map();
    this.usage = undefined;
    this.timings = undefined;
    this.finishReason = null;
  }

  /** @returns {Array<{type:'text', value:string}>} */
  push(chunk) {
    const parts = [];
    if (chunk.usage) this.usage = chunk.usage;
    if (chunk.timings) this.timings = chunk.timings;
    const choice = chunk.choices && chunk.choices[0];
    if (!choice) return parts;
    const delta = choice.delta || {};
    if (typeof delta.content === 'string' && delta.content.length) {
      parts.push({ type: 'text', value: delta.content });
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        const i = typeof tc.index === 'number' ? tc.index : 0;
        const cur = this.calls.get(i) || { id: '', name: '', args: '' };
        if (tc.id) cur.id = tc.id;
        if (tc.function) {
          if (tc.function.name) cur.name += tc.function.name;
          if (typeof tc.function.arguments === 'string') cur.args += tc.function.arguments;
        }
        this.calls.set(i, cur);
      }
    }
    if (choice.finish_reason) this.finishReason = choice.finish_reason;
    return parts;
  }

  /**
   * @param {Map<string, object>} [schemas] tool name -> JSON schema (for argument type coercion)
   * @returns {Array<{type:'tool', callId:string, name:string, input:object}>}
   */
  flush(schemas) {
    const out = [];
    for (const [, c] of [...this.calls.entries()].sort((a, b) => a[0] - b[0])) {
      if (!c.name) continue;
      let input = {};
      if (c.args.trim().length) {
        try {
          input = JSON.parse(c.args);
        } catch {
          throw new Error(`Model produced invalid JSON arguments for tool "${c.name}": ${c.args.slice(0, 200)}`);
        }
      }
      out.push({ type: 'tool', callId: c.id || genCallId(), name: c.name, input: coerceArgs(input, schemas && schemas.get(c.name)) });
    }
    this.calls.clear();
    return out;
  }
}

module.exports = {
  ROLE_USER,
  ROLE_ASSISTANT,
  toOpenAIMessages,
  toOpenAITools,
  readSSE,
  StreamAccumulator,
  toolResultText,
  normalizeSchema,
  coerceArgs,
};
