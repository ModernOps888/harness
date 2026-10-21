'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { toOpenAIMessages, toOpenAITools, StreamAccumulator, readSSE } = require('../lib/openai');

// Shape-compatible stand-ins for vscode.LanguageModel*Part (duck-typed by the library)
const text = (value) => ({ value });
const call = (callId, name, input) => ({ callId, name, input });
const result = (callId, content) => ({ callId, content });

test('assistant tool call + user tool result round-trip to OpenAI shape', () => {
  const msgs = toOpenAIMessages([
    { role: 1, content: [text('List the files')] },
    { role: 2, content: [text('Looking.'), call('c1', 'list_dir', { path: '.' })] },
    { role: 1, content: [result('c1', [text('a.rs\nb.rs')])] },
  ]);
  assert.deepEqual(msgs, [
    { role: 'user', content: 'List the files' },
    {
      role: 'assistant',
      content: 'Looking.',
      tool_calls: [{ id: 'c1', type: 'function', function: { name: 'list_dir', arguments: '{"path":"."}' } }],
    },
    { role: 'tool', tool_call_id: 'c1', content: 'a.rs\nb.rs' },
  ]);
});

test('assistant turn with only a tool call has null content', () => {
  const [m] = toOpenAIMessages([{ role: 2, content: [call('x', 'f', {})] }]);
  assert.equal(m.content, null);
  assert.equal(m.tool_calls[0].function.arguments, '{}');
});

test('tool results precede accompanying user text', () => {
  const msgs = toOpenAIMessages([{ role: 1, content: [text('thanks, continue'), result('c9', [text('ok')])] }]);
  assert.equal(msgs[0].role, 'tool');
  assert.equal(msgs[1].role, 'user');
});

test('tools map to OpenAI function schema with default empty schema', () => {
  const tools = toOpenAITools([
    { name: 'read_file', description: 'Read', inputSchema: { type: 'object', properties: { p: { type: 'string' } } } },
    { name: 'noargs', description: 'x' },
  ]);
  assert.equal(tools[0].type, 'function');
  assert.equal(tools[0].function.parameters.properties.p.type, 'string');
  assert.deepEqual(tools[1].function.parameters, { type: 'object', properties: {} });
  assert.equal(toOpenAITools([]), undefined);
});

test('stream accumulator assembles fragmented tool-call arguments and text', () => {
  const acc = new StreamAccumulator();
  const t1 = acc.push({ choices: [{ delta: { content: 'Hi ' } }] });
  acc.push({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'abc', function: { name: 'read_file', arguments: '{"pa' } }] } }] });
  acc.push({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: 'th":"x.rs"}' } }] } }] });
  acc.push({ choices: [{ delta: {}, finish_reason: 'tool_calls' }], usage: { prompt_tokens: 10, completion_tokens: 5 } });
  assert.deepEqual(t1, [{ type: 'text', value: 'Hi ' }]);
  const calls = acc.flush();
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], { type: 'tool', callId: 'abc', name: 'read_file', input: { path: 'x.rs' } });
  assert.equal(acc.finishReason, 'tool_calls');
  assert.equal(acc.usage.prompt_tokens, 10);
});

test('stream accumulator generates ids and rejects invalid argument JSON', () => {
  const a = new StreamAccumulator();
  a.push({ choices: [{ delta: { tool_calls: [{ index: 0, function: { name: 'f', arguments: '{}' } }] } }] });
  assert.match(a.flush()[0].callId, /^call_/);
  const b = new StreamAccumulator();
  b.push({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'z', function: { name: 'f', arguments: '{bad' } }] } }] });
  assert.throws(() => b.flush(), /invalid JSON arguments/);
});

test('readSSE reassembles events split across arbitrary chunk boundaries', async () => {
  const enc = new TextEncoder();
  const payload = 'data: {"a":1}\n\ndata: {"b":2}\n\ndata: [DONE]\n\n';
  async function* chunks() {
    for (let i = 0; i < payload.length; i += 5) yield enc.encode(payload.slice(i, i + 5));
  }
  const seen = [];
  await readSSE(chunks(), (d) => seen.push(d));
  assert.deepEqual(seen, ['{"a":1}', '{"b":2}', '[DONE]']);
});

const { normalizeSchema, coerceArgs } = require('../lib/openai');

test('schemas without properties / non-object schemas are normalised for the Jinja template', () => {
  assert.deepEqual(normalizeSchema({ type: 'object' }), { type: 'object', properties: {} });
  assert.deepEqual(normalizeSchema(undefined), { type: 'object', properties: {} });
  assert.deepEqual(normalizeSchema({ type: 'string' }), { type: 'object', properties: {} });
  assert.deepEqual(normalizeSchema({ type: 'object', properties: { a: { type: 'string' } }, required: 'a' }).required, undefined);
  const tools = toOpenAITools([{ name: 'x', description: 'd', inputSchema: { type: 'object' } }]);
  assert.deepEqual(tools[0].function.parameters, { type: 'object', properties: {} });
});

test('string tool arguments are coerced to the schema types (Qwen emits raw strings)', () => {
  const schema = { properties: { n: { type: 'integer' }, f: { type: 'boolean' }, a: { type: 'array' }, s: { type: 'string' }, o: { type: 'object' } } };
  const out = coerceArgs({ n: '42', f: 'True', a: '[1,2]', s: '42', o: '{"k":1}' }, schema);
  assert.deepEqual(out, { n: 42, f: true, a: [1, 2], s: '42', o: { k: 1 } });
  assert.deepEqual(coerceArgs({ n: 'abc' }, schema), { n: 'abc' });
});

test('flush applies coercion using the tool schema map', () => {
  const acc = new StreamAccumulator();
  acc.push({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'a', function: { name: 'f', arguments: '{"line":"7"}' } }] } }] });
  const [c] = acc.flush(new Map([['f', { type: 'object', properties: { line: { type: 'integer' } } }]]));
  assert.deepEqual(c.input, { line: 7 });
});

test('system-role messages are mapped to a system message', () => {
  const msgs = toOpenAIMessages([{ role: 3, content: [text('be terse')] }, { role: 1, content: [text('hi')] }]);
  assert.deepEqual(msgs, [{ role: 'system', content: 'be terse' }, { role: 'user', content: 'hi' }]);
});
