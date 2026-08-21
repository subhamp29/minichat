import torch, types
import train_lora as T
from transformers import AutoTokenizer

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))
m.config.attn_implementation = "eager"
for mod in m.modules():
    if hasattr(mod, "config"):
        try:
            mod.config.attn_implementation = "eager"
        except Exception:
            pass
m.eval()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True, trust_remote_code=True)
ids = tok("hello world", return_tensors="pt").input_ids
with torch.no_grad():
    out = m(ids)
print("EAGER attn logits: max", float(out.logits.max()), "min", float(out.logits.min()),
      "mean", float(out.logits.float().mean()), "nan", bool(torch.isnan(out.logits).any()),
      "inf", bool(torch.isinf(out.logits).any()), flush=True)
