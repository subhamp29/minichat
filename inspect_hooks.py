import torch, types
import train_lora as T
from transformers import AutoTokenizer

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))
m.eval()

norms = {}
def mk(i):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        norms[i] = float(h.float().norm())
    return hook
for i, layer in enumerate(m.model.layers):
    layer.register_forward_hook(mk(i))

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True, trust_remote_code=True)
ids = tok("hello world", return_tensors="pt").input_ids
with torch.no_grad():
    out = m(ids)
for i in range(min(4, len(norms))):
    print(f"layer {i} out norm:", f"{norms[i]:.4e}", flush=True)
print("layer 23 out norm:", f"{norms.get(23, float('nan')):.4e}", flush=True)
print("logits max:", float(out.logits.max()), flush=True)
