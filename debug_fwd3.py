import faulthandler, torch, psutil
faulthandler.enable()
torch.set_num_threads(4)
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
import train_lora as T

def rss():
    return round(psutil.Process().memory_info().rss / 1e6, 1)

# mmap read-only base (load_base_model_cpu attaches 0-commit file-backed weights)
m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False)) if False else None
import types
m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))
print("base mmap'd; rss MB:", rss(), flush=True)

m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=T.TARGET_MODULES,
                                 lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
print("after PEFT; trainable params:", m.print_trainable_parameters(), "rss MB:", rss(), flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True)
ids = tok("hello world, a test sentence here", return_tensors="pt").input_ids
m.train()
print("forward+backward (mmap base)...", flush=True)
out = m(input_ids=ids, labels=ids.clone())
print("loss:", float(out.loss.detach()), "rss MB:", rss(), flush=True)
out.loss.backward()
print("backward OK; rss MB:", rss(), flush=True)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
opt.step()
print("optim.step OK; rss MB:", rss(), flush=True)
