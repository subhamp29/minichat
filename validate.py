import torch, types
import train_lora as T

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True, trust_remote_code=True)
ids = tok("hello world", return_tensors="pt").input_ids
m.eval()
with torch.no_grad():
    out = m(ids)
lg = out.logits
print("logits dtype", lg.dtype, "shape", tuple(lg.shape), flush=True)
print("max", float(lg.max()), "min", float(lg.min()), "mean", float(lg.float().mean()), flush=True)
print("nan", bool(torch.isnan(lg).any()), "inf", bool(torch.isinf(lg).any()), flush=True)
# sample a few logits
print("sample[0,0,:5]", lg[0, 0, :5].tolist(), flush=True)
