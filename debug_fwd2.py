import faulthandler, os, json, struct, torch, psutil
faulthandler.enable()
torch.set_num_threads(4)
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from train_lora import _SF_DT, TARGET_MODULES

def rss():
    return round(psutil.Process().memory_info().rss / 1e6, 1)

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
SF = os.path.join(SNAP, "model.safetensors")

with open(SF, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))

cfg = AutoConfig.from_pretrained(SNAP)
with init_empty_weights():
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)

fh = open(SF, "rb")
loaded = 0
with fh:
    for name, info in hdr.items():
        if "data_offsets" not in info:
            continue
        dt, esz = _SF_DT[info["dtype"]]
        start, end = info["data_offsets"]
        fh.seek(start)
        buf = bytearray(fh.read(end - start))
        t = torch.frombuffer(buf, dtype=dt, count=(end - start) // esz).reshape(info["shape"])
        parts = name.split(".")
        parent = m.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else m
        setattr(parent, parts[-1], torch.nn.Parameter(t, requires_grad=False))
        loaded += 1
m.tie_weights()
for p in m.parameters():
    p.requires_grad_(False)
print("loaded", loaded, "tensors to RAM; rss MB:", rss(), flush=True)

m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET_MODULES,
                                 lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
print("trainable:", m.print_trainable_parameters(), "; rss MB:", rss(), flush=True)

tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True)
ids = tok("hello world, a test sentence here", return_tensors="pt").input_ids
m.train()
print("forward+backward", flush=True)
out = m(input_ids=ids, labels=ids.clone())
print("loss:", float(out.loss), "rss MB:", rss(), flush=True)
out.loss.backward()
print("backward OK; rss MB:", rss(), flush=True)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
opt.step()
print("step OK; rss MB:", rss(), flush=True)
