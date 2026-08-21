import torch, types
import train_lora as T
from transformers import AutoTokenizer

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))
m.eval()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True, trust_remote_code=True)
ids = tok("hello world", return_tensors="pt").input_ids
print("input_ids", ids.tolist(), flush=True)

with torch.no_grad():
    embs = m.model.embed_tokens(ids)
    print("embed norm (rms):", float((embs.float().pow(2).mean().sqrt())),
          "absmax", float(embs.float().abs().max()), flush=True)

    h = embs
    for i, layer in enumerate(m.model.layers):
        res = layer(h)
        h = res[0] if isinstance(res, (tuple, list)) else res
        rms = float((h.float().pow(2).mean().sqrt()).squeeze()) if h.dim() > 1 else 0.0
        print(f"layer {i:2d} output rms={rms:.4e} absmax={float(h.float().abs().max()):.4e}", flush=True)
        if i >= 4:
            break

    fn = m.model.norm(h)
    print("final norm rms:", float((fn.float().pow(2).mean().sqrt())), flush=True)
    logits = m.lm_head(fn)
    print("logits rms:", float((logits.float().pow(2).mean().sqrt())),
          "max", float(logits.max()), flush=True)
print("DONE", flush=True)
