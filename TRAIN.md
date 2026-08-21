# Training Bhavyam AI for accuracy (LoRA)

This trains a cheap **LoRA adapter** on top of `Qwen2.5-0.5B-Instruct` so the
local model becomes accurate across formats (arithmetic, comparisons, code,
factual, reasoning). The base weights stay frozen, so it runs on a CPU laptop.

> You only need to do this if you want a *model-level* accuracy boost. The
> `app.py` fixes (lower temperature, gated web search, output cleaning) already
> handle most wrong-answer causes without any training.

## 1. Install dependencies (one-time)

```bash
pip install torch transformers peft datasets accelerate
```

- **GPU**: also install `bitsandbytes` and run with `--load_in_4bit` (fast).
- **CPU**: `torch` CPU build is enough; training is slow but works for 0.5B.

## 2. Files

- `finetune_dataset.py` — curated, high-quality Q/A examples (the "accuracy"
  labels). Edit/extend the `PAIRS` list to add more, then `python
  finetune_dataset.py` to (re)generate `finetune_dataset.jsonl`.
- `finetune_dataset.jsonl` — the generated dataset (92 examples).
- `train_lora.py` — the LoRA training script.

## 3. Run training

CPU (slow):
```bash
python train_lora.py --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset finetune_dataset.jsonl --output_dir ./lora-accurate \
    --epochs 5 --per_device_train_batch_size 1 --gradient_accumulation_steps 8
```

GPU (fast, 4-bit):
```bash
python train_lora.py --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset finetune_dataset.jsonl --output_dir ./lora-accurate \
    --epochs 3 --per_device_train_batch_size 4 --load_in_4bit
```

This produces:
- `./lora-accurate/adapter_model.safetensors` (the LoRA weights)
- `./lora-accurate/tokenizer.*`

## 4. Use the adapter with MiniChat/app.py

The app loads GGUF models via `llama-cpp-python`. You have two options:

### Option A — runtime LoRA (lightweight; keep base GGUF, load adapter at runtime)
`llama-cpp-python`'s `Llama()` accepts a GGUF LoRA adapter:

```python
from llama_cpp import Llama
llm = Llama(
    model_path="qwen2.5-0.5b-q4_k_m.gguf",      # base GGUF
    lora_base="qwen2.5-0.5b-q4_k_m.gguf",
    lora_adapter="adapter.ggml.lora.gguf",        # converted from the peft adapter
    lora_scale=1.0,
    n_ctx=4096, n_threads=os.cpu_count(), n_gpu_layers=0, n_batch=128,
)
```

Convert the peft adapter to a GGUF LoRA with llama.cpp tooling:
```bash
# from a llama.cpp checkout:
python convert_lora_to_gguf.py --adapter_dir ./lora-accurate \
    --model_base qwen2.5-0.5b-q4_k_m.gguf --outfile adapter.ggml.lora.gguf
```
Then edit `app.py` to pass `lora_base`/`lora_adapter` into the `Llama(...)`
call (see `load_model`).

### Option B — baked-in GGUF (recommended; simplest for the app)
Merge the LoRA into the base model and re-quantize to a single GGUF, then add it
to `MODEL_OPTIONS` like the other models so the app downloads/loads it normally:

```bash
# 1) merge into a dense HF checkpoint
python train_lora.py ... --merge     # writes ./lora-accurate/merged_dense/

# 2) convert HF -> GGUF with llama.cpp
python /path/to/llama.cpp/convert_hf_to_gguf.py \
    --model_dir ./lora-accurate/merged_dense \
    --outfile ./qwen2.5-0.5b-accurate.gguf

# 3) quantize (use Q4_K_M to stay small)
python /path/to/llama.cpp/quantize \
    ./qwen2.5-0.5b-accurate.gguf ./qwen2.5-0.5b-accurate-q4_k_m.gguf Q4_K_M

# 4) place into the cache the app reads, e.g.:
#    .model_cache/models--local--qwen2.5-0.5b-accurate/
#    (or upload to a HuggingFace repo and add an entry to MODEL_OPTIONS in app.py)
```

## 5. What this fixes
The dataset deliberately over-weights the formats the base 0.5B model gets wrong
today: comparisons/arithmetic (`5 < 3` / `5 > 3`), algebra, code correctness,
and factual recall. Training steers the model to answer these correctly instead
of confidently wrong.

## Notes
- LoRA rank `r=16` keeps the adapter tiny (~1–2 MB) and training fast.
- CPU: ~5–15 minutes for 5 epochs on 92 examples (0.5B model).
- To update accuracy later, edit `finetune_dataset.py`, rerun it, and resume/fine-tune again.
