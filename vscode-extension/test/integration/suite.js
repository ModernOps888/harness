'use strict';
/* Runs inside the VS Code extension host (real vscode API). */
const vscode = require('vscode');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const log = (...a) => console.log('[harness-it]', ...a);

async function collect(model, messages, options, token) {
  const res = await model.sendRequest(messages, options, token);
  let text = '';
  const calls = [];
  for await (const part of res.stream) {
    if (part instanceof vscode.LanguageModelTextPart) text += part.value;
    else if (part instanceof vscode.LanguageModelToolCallPart) calls.push(part);
  }
  return { text, calls };
}

async function run() {
  const token = new vscode.CancellationTokenSource().token;
  const ext = vscode.extensions.all.find((e) => e.packageJSON && e.packageJSON.name === 'harness-vscode');
  assert.ok(ext, 'harness-vscode extension must be loaded');
  await ext.activate();

  // 1. Model discovery through the native vscode.lm API
  let models = [];
  for (let i = 0; i < 20 && !models.length; i++) {
    models = await vscode.lm.selectChatModels({ vendor: 'harness' });
    if (!models.length) await new Promise((r) => setTimeout(r, 1000));
  }
  assert.ok(models.length >= 1, 'a HARNESS model must be discoverable via vscode.lm.selectChatModels');
  const model = models[0];
  log('model', model.id, 'maxInputTokens', model.maxInputTokens, 'maxOutputTokens', model.maxOutputTokens);
  assert.ok(model.capabilities.supportsToolCalling, 'model must advertise native tool calling');

  // 2. Plain streamed chat
  const plain = await collect(
    model,
    [vscode.LanguageModelChatMessage.User('Reply with exactly the single word: PONG')],
    { modelOptions: { max_tokens: 16 } },
    token
  );
  log('plain reply:', JSON.stringify(plain.text));
  assert.match(plain.text, /PONG/i);

  // 3. Native multi-tool agent loop: the model picks tools, VS Code API carries calls/results.
  const ws = process.env.HARNESS_TEST_WORKSPACE;
  const secret = `HX-${Math.floor(Math.random() * 1e6)}`;
  fs.writeFileSync(path.join(ws, 'config.txt'), `build_id=${secret}\nowner=harness\n`);
  fs.writeFileSync(path.join(ws, 'notes.md'), '# notes\nnothing here\n');

  const tools = [
    {
      name: 'list_dir',
      description: 'List the files in a directory of the workspace. Use "." for the workspace root.',
      inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
    },
    {
      name: 'read_file',
      description: 'Read a text file from the workspace and return its contents.',
      inputSchema: { type: 'object', properties: { path: { type: 'string', description: 'relative file path' } }, required: ['path'] },
    },
  ];
  const impl = {
    list_dir: (a) => fs.readdirSync(path.join(ws, a.path || '.')).join('\n'),
    read_file: (a) => fs.readFileSync(path.join(ws, a.path), 'utf8'),
  };

  const messages = [
    vscode.LanguageModelChatMessage.User(
      'You are a coding agent with tools. List the workspace directory, find the file that contains a build_id, read it, and tell me the build_id value. Use the tools; do not guess.'
    ),
  ];
  const used = [];
  let finalText = '';
  for (let turn = 0; turn < 6; turn++) {
    const { text, calls } = await collect(model, messages, { tools, toolMode: vscode.LanguageModelChatToolMode.Auto, modelOptions: { max_tokens: 400 } }, token);
    log(`turn ${turn}: text=${JSON.stringify(text.slice(0, 80))} calls=${calls.map((c) => c.name + JSON.stringify(c.input)).join(', ')}`);
    if (!calls.length) {
      finalText = text;
      break;
    }
    messages.push(vscode.LanguageModelChatMessage.Assistant([...(text ? [new vscode.LanguageModelTextPart(text)] : []), ...calls]));
    const results = calls.map((c) => {
      used.push(c.name);
      let out;
      try {
        out = impl[c.name](c.input);
      } catch (e) {
        out = `ERROR: ${e.message}`;
      }
      return new vscode.LanguageModelToolResultPart(c.callId, [new vscode.LanguageModelTextPart(out)]);
    });
    messages.push(vscode.LanguageModelChatMessage.User(results));
  }
  log('tools used:', used.join(' -> '));
  log('final answer:', JSON.stringify(finalText));
  assert.ok(used.includes('read_file'), 'model must call read_file');
  assert.ok(used.length >= 2, 'model must chain at least two distinct tool calls (multi-tool)');
  assert.ok(finalText.includes(secret), `final answer must contain the secret ${secret}`);

  // 4. Informational: which native chat tools does this VS Code expose?
  log('vscode.lm.tools registered:', vscode.lm.tools.length, vscode.lm.tools.slice(0, 12).map((t) => t.name).join(', '));
}

module.exports = { run };
