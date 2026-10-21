# HARNESS Local Models for VS Code Native Chat

Registers local **HARNESS** models (`Qwen3-Coder-30B-A3B` 32k/64k Agent Engine, `Qwen2.5-Coder-32B`, `Qwen2.5-Coder-14B`) directly into VS Code's built-in **Chat / Agent Mode** via `vscode.lm.registerLanguageModelChatProvider('harness', ...)`.

## Features
- **Native VS Code Chat & Agent Mode**: Select `HARNESS: qwen3-coder-30b-a3b` directly from the built-in VS Code Chat model picker.
- **Full Multi-Tool Calling**: Supports `tools`, `tool_choice`, parallel tool calls, `LanguageModelToolCallPart`, and `LanguageModelToolResultPart` in multi-turn agent loops.
- **Long-Context GPU Prefill (`32,768` default, up to `65,536`)**: Queries `/props` on the HARNESS server to advertise the exact context window (`maxInputTokens = n_ctx - maxOutputTokens`).
- **One-Command Engine Launch**: Run `HARNESS: Start Local Coder Engine (llama-server + proxy)` or `HARNESS: Check Local Server Status & Speed` from the Command Palette.
