import torch, types, psutil
torch.set_num_threads(4)
import train_lora as T
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

def rss():
    return round(psutil.Process().memory_info().rss / 1e6, 1)

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))
print("loaded mmap bf16; rss MB:", rss(), flush=True)
m = m.to(torch.float16)  # writable fp16 RAM copy (~942 MB)
print("cast to fp16; rss MB:", rss(), flush=True)
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=T.TARGET_MODULES,
                                 lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
print("PEFT; trainable:", m.print_trainable_parameters(), "rss MB:", rss(), flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True, trust_remote_code=True)
ids = tok("hello world test", return_tensors="pt").input_ids
m.train()
print("forward+backward (fp16)...", flush=True)
out = m(input_ids=ids, labels=ids.clone())
print("loss:", float(out.loss), "logits max:", float(out.logits.max()), "rss MB:", rss(), flush=True)
out.loss.backward()
print("backward OK; rss MB:", rss(), flush=True)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
opt.step()
print("optim.step OK; rss MB:", rss(), flush=True)
