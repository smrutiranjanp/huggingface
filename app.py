import csv
import ipaddress
import json
import os
import socket
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import gradio as gr


DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"
MAX_BATCH_ROWS = 200
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MODEL_CACHE = {}
MODEL_CACHE_LOCK = threading.Lock()


def _validate_endpoint(endpoint):
    """Accept public HTTPS API endpoints and reject local/private destinations."""
    endpoint = (endpoint or "").strip().rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter a public HTTPS API base URL (for example, https://integrate.api.nvidia.com/v1).")
    if parsed.port not in (None, 443):
        raise ValueError("Only the standard HTTPS port (443) is supported.")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in {"localhost", "localhost.localdomain", "metadata.google.internal"} or hostname.endswith(".localhost"):
        raise ValueError("Local and metadata endpoints are not allowed.")
    try:
        addresses = [ipaddress.ip_address(hostname)] if not hostname.replace(".", "").isdigit() else []
        if not addresses:
            addresses = [ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)]
        if any(not address.is_global for address in addresses):
            raise ValueError("Private, local, and reserved network destinations are not allowed.")
    except socket.gaierror as exc:
        raise ValueError("The endpoint hostname could not be resolved.") from exc
    except ValueError:
        raise
    return endpoint


def _chat_url(base_url):
    return base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"


def _estimate_tokens(text):
    return max(1, (len(text) + 3) // 4) if text else 0


def _api_inference(config, prompt, system_prompt, max_tokens, temperature):
    endpoint = _validate_endpoint(config.get("endpoint"))
    model = (config.get("model") or "").strip()
    if not model:
        raise ValueError("Enter a model name or ID.")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = (config.get("api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(_chat_url(endpoint), data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # Do not echo request headers or secrets in UI error messages.
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from None
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not complete the API request: {exc}") from None
    elapsed = time.perf_counter() - started
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("The endpoint response did not contain a chat completion.") from None
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    usage = result.get("usage") or {}
    return {
        "text": str(text),
        "latency": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", _estimate_tokens(prompt + system_prompt)),
        "completion_tokens": usage.get("completion_tokens", _estimate_tokens(str(text))),
        "token_source": "reported by API" if usage else "estimated",
    }


def _load_local_model(model_id):
    with MODEL_CACHE_LOCK:
        if model_id not in MODEL_CACHE:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Local inference requires the torch and transformers packages.") from exc
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                if not torch.cuda.is_available():
                    model.to("cpu")
                model.eval()
                MODEL_CACHE[model_id] = (tokenizer, model)
            except Exception as exc:
                raise RuntimeError(f"Could not load local model {model_id!r}: {exc}") from None
        return MODEL_CACHE[model_id]


def _local_inference(config, prompt, system_prompt, max_tokens, temperature):
    model_id = (config.get("model") or "").strip()
    if not model_id:
        raise ValueError("Enter a Hugging Face model ID or local model path.")
    import torch

    tokenizer, model = _load_local_model(model_id)
    conversation = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:" if system_prompt else f"User: {prompt}\n\nAssistant:"
    inputs = tokenizer(conversation, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    generation = {"max_new_tokens": int(max_tokens), "do_sample": float(temperature) > 0}
    if float(temperature) > 0:
        generation["temperature"] = float(temperature)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **generation)
    elapsed = time.perf_counter() - started
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "text": text,
        "latency": elapsed,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "completion_tokens": int(generated.shape[0]),
        "token_source": "tokenizer",
    }


def _infer(config, prompt, system_prompt, max_tokens, temperature):
    if config["mode"] == "Local Transformers":
        return _local_inference(config, prompt, system_prompt, max_tokens, temperature)
    return _api_inference(config, prompt, system_prompt, max_tokens, temperature)


def _run_pair(prompt, system_prompt, max_tokens, temperature, left_config, right_config):
    if not (prompt or "").strip():
        raise gr.Error("Enter a prompt before running the comparison.")
    with ThreadPoolExecutor(max_workers=2) as pool:
        left_future = pool.submit(_infer, left_config, prompt, system_prompt, max_tokens, temperature)
        right_future = pool.submit(_infer, right_config, prompt, system_prompt, max_tokens, temperature)
        results = []
        for future in (left_future, right_future):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"error": str(exc)})
    return results


def _result_text(result):
    return result.get("text", f"Inference error: {result.get('error', 'unknown error')}")


def _metrics_markdown(results, labels):
    lines = ["| Metric | " + " | ".join(labels) + " |", "|:--|" + "|".join(["--:"] * len(labels)) + "|"]
    for title, key, formatter in (
        ("Latency", "latency", lambda value: f"{value:.2f} s"),
        ("Input tokens", "prompt_tokens", str),
        ("Output tokens", "completion_tokens", str),
    ):
        values = [formatter(result[key]) if key in result else "—" for result in results]
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    sources = [result.get("token_source", "—") for result in results]
    lines.append("| Token counts | " + " | ".join(sources) + " |")
    return "\n".join(lines)


def compare(prompt, system_prompt, max_tokens, temperature, left_mode, left_model, left_endpoint, left_key,
            right_mode, right_model, right_endpoint, right_key):
    left_config = {"mode": left_mode, "model": left_model, "endpoint": left_endpoint, "api_key": left_key}
    right_config = {"mode": right_mode, "model": right_model, "endpoint": right_endpoint, "api_key": right_key}
    results = _run_pair(prompt, system_prompt, max_tokens, temperature, left_config, right_config)
    return (
        _result_text(results[0]), _result_text(results[1]),
        _metrics_markdown(results, [left_model or "Model A", right_model or "Model B"]),
    )


def _read_test_set(file_path):
    if not file_path:
        raise gr.Error("Upload a CSV or JSONL test set first.")
    if os.path.getsize(file_path) > MAX_UPLOAD_BYTES:
        raise gr.Error("The test set must be 2 MB or smaller.")
    rows = []
    try:
        if file_path.lower().endswith(".jsonl"):
            with open(file_path, encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if line.strip():
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise ValueError(f"JSONL line {line_number} must be an object.")
                        rows.append(item)
        elif file_path.lower().endswith(".csv"):
            with open(file_path, newline="", encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
        else:
            raise ValueError("Upload a .csv or .jsonl file.")
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, ValueError) as exc:
        raise gr.Error(f"Could not read the test set: {exc}") from exc
    if not rows:
        raise gr.Error("The test set has no data rows.")
    if len(rows) > MAX_BATCH_ROWS:
        raise gr.Error(f"The test set is limited to {MAX_BATCH_ROWS} rows per run.")
    return rows


def _field(row, names):
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _score(output, expected):
    if not expected:
        return ""
    normalize = lambda value: " ".join(value.casefold().split())
    return int(normalize(output) == normalize(expected))


def run_batch(file_path, system_prompt, max_tokens, temperature, left_mode, left_model, left_endpoint, left_key,
              right_mode, right_model, right_endpoint, right_key):
    rows = _read_test_set(file_path)
    configs = [
        {"mode": left_mode, "model": left_model, "endpoint": left_endpoint, "api_key": left_key},
        {"mode": right_mode, "model": right_model, "endpoint": right_endpoint, "api_key": right_key},
    ]
    output_rows = []
    for index, row in enumerate(rows, 1):
        prompt = _field(row, ("prompt", "input", "text", "question"))
        expected = _field(row, ("expected", "target", "reference", "expected_output", "answer"))
        if not prompt:
            raise gr.Error(f"Row {index} is missing a prompt/input/text/question value.")
        results = _run_pair(prompt, system_prompt or "", max_tokens, temperature, configs[0], configs[1])
        output_rows.append({
            "prompt": prompt,
            "expected": expected,
            "model_a_output": _result_text(results[0]),
            "model_a_latency_s": round(results[0]["latency"], 3) if "latency" in results[0] else "",
            "model_a_input_tokens": results[0].get("prompt_tokens", ""),
            "model_a_output_tokens": results[0].get("completion_tokens", ""),
            "model_a_exact_match": _score(results[0].get("text", ""), expected),
            "model_b_output": _result_text(results[1]),
            "model_b_latency_s": round(results[1]["latency"], 3) if "latency" in results[1] else "",
            "model_b_input_tokens": results[1].get("prompt_tokens", ""),
            "model_b_output_tokens": results[1].get("completion_tokens", ""),
            "model_b_exact_match": _score(results[1].get("text", ""), expected),
        })
    fields = list(output_rows[0].keys())
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", suffix=".csv", delete=False) as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
        result_path = target.name
    scored = sum(bool(row["expected"]) for row in output_rows)
    summary = f"Completed **{len(output_rows)}** prompts. Exact-match scores are included for {scored} row(s) with expected answers."
    return result_path, summary


def model_controls(label, default_model):
    gr.Markdown(f"### {label}")
    mode = gr.Dropdown(["OpenAI-compatible API", "Local Transformers"], value="OpenAI-compatible API", label="Inference provider")
    model = gr.Textbox(value=default_model, label="Model name / Hugging Face model ID", placeholder="organization/model-name")
    endpoint = gr.Textbox(value=DEFAULT_ENDPOINT, label="API base URL (HTTPS)", info="Used for OpenAI-compatible endpoints such as NVIDIA NIM.")
    key = gr.Textbox(label="API key (optional)", type="password", placeholder="Enter a provider key")
    return mode, model, endpoint, key


with gr.Blocks(title="Model Playground & Evaluator", theme=gr.themes.Soft(), css="""
    .main-title { text-align: center; margin-bottom: 0.25rem; }
    .subtitle { text-align: center; color: #64748b; margin-bottom: 1.5rem; }
    .model-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 1rem; }
""") as demo:
    gr.Markdown("# 🧪 Model Playground & Evaluator", elem_classes=["main-title"])
    gr.Markdown("Compare model responses side by side, then run a CSV or JSONL test set and download the results.", elem_classes=["subtitle"])

    with gr.Tab("Playground"):
        with gr.Row():
            with gr.Column(elem_classes=["model-card"]):
                left = model_controls("Model A", "meta/llama-3.1-8b-instruct")
            with gr.Column(elem_classes=["model-card"]):
                right = model_controls("Model B", "meta/llama-3.1-8b-instruct")
        prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Enter a prompt to send to both models…")
        system_prompt = gr.Textbox(label="System prompt (optional)", lines=2, value="You are a helpful assistant.")
        with gr.Row():
            max_tokens = gr.Slider(minimum=1, maximum=2048, value=256, step=1, label="Maximum new tokens")
            temperature = gr.Slider(minimum=0, maximum=2, value=0.2, step=0.1, label="Temperature")
        run = gr.Button("Run comparison", variant="primary")
        with gr.Row():
            output_a = gr.Textbox(label="Model A response", lines=12, interactive=False)
            output_b = gr.Textbox(label="Model B response", lines=12, interactive=False)
        metrics = gr.Markdown("Latency and token usage will appear here after inference.")
        run.click(compare, inputs=[prompt, system_prompt, max_tokens, temperature, *left, *right], outputs=[output_a, output_b, metrics])

    with gr.Tab("Batch evaluator"):
        gr.Markdown("Upload a **CSV** or **JSONL** file. Each row needs a `prompt`, `input`, `text`, or `question` field. An optional `expected`, `target`, `reference`, `expected_output`, or `answer` field enables normalized exact-match scoring.")
        batch_file = gr.File(label="Test set", file_types=[".csv", ".jsonl"], type="filepath")
        batch_system = gr.Textbox(label="System prompt (optional)", value="You are a helpful assistant.")
        with gr.Row():
            batch_max_tokens = gr.Slider(minimum=1, maximum=2048, value=256, step=1, label="Maximum new tokens")
            batch_temperature = gr.Slider(minimum=0, maximum=2, value=0.2, step=0.1, label="Temperature")
        batch_run = gr.Button("Run batch evaluation", variant="primary")
        batch_status = gr.Markdown()
        download = gr.File(label="Download scored results")
        batch_run.click(
            run_batch,
            inputs=[batch_file, batch_system, batch_max_tokens, batch_temperature, *left, *right],
            outputs=[download, batch_status],
        )

if __name__ == "__main__":
    demo.queue().launch()
