---
title: Model Playground & Evaluator
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Compare and evaluate language models side by side
---

# Model Playground & Evaluator

A Gradio app for comparing base, fine-tuned, or hosted language models. Run prompts side by side and inspect latency and token usage, or upload a CSV/JSONL test set and download a scored CSV.

## Inference providers

- **OpenAI-compatible API:** Supports providers such as NVIDIA NIM. Set the API base URL and provide a key if your provider requires one. The app accepts public HTTPS endpoints on the standard port.
- **Local Transformers:** Load a Hugging Face causal language model by model ID or local path. Models are cached for the current app process; use a Space with sufficient RAM/VRAM.

API keys are entered as password fields and are used only for the inference request. They are not written to the result file.

## Batch test format

CSV files need a header row. JSONL files need one JSON object per line. Use `prompt` (or `input`, `text`, `question`) for the input and optionally `expected` (or `target`, `reference`, `expected_output`, `answer`) for the reference answer. Exact-match scores normalize case and whitespace. Batch uploads are limited to 2 MB and 200 rows.
