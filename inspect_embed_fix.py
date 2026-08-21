import os, struct, json, mmap, logging
logging.basicConfig(level=logging.WARNING)
import torch
torch.set_num_threads(4)

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
ST_PATH = os.path.join(SNAP, "model.safetensors")
from train_lora import _SF_DT

def p(*a): print(*a, flush=True)

with open(ST_PATH, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len).decode("utf-8"))
p(f"header_len (JSON bytes) = {header_len}; data section starts at file byte {8 + header_len}")

fh = open(ST_PATH, "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
DATA_BASE = 8 + header_len  # absolute file offset where tensor data begins

# embed with CORRECT absolute offset
k = "model.embed_tokens.weight"
info = header[k]
start, end = info["data_offsets"]
dt, esz = _SF_DT[info["dtype"]]
n = (end - start) // esz
t_correct = torch.frombuffer(mm, dtype=dt, count=n, offset=start + DATA_BASE).reshape(info["shape"])
p(f"embed CORRECT-offset dtype={t_correct.dtype} shape={tuple(t_correct.shape)}")
p(f"embed min={float(t_correct.min())} max={float(t_correct.max())} mean={float(t_correct.float().mean())} std={float(t_correct.float().std())}")
p(f"embed row0[:8]={[round(x,4) for x in t_correct[0,:8].tolist()]}")

# also confirm q_proj still sane with correct offset
k2 = "model.layers.0.self_attn.q_proj.weight"
info2 = header[k2]
s2,e2 = info2["data_offsets"]; dt2,esz2=_SF_DT[info2["dtype"]]
n2=(e2-s2)//esz2
tq = torch.frombuffer(mm, dtype=dt2, count=n2, offset=s2+DATA_BASE).reshape(info2["shape"])
p(f"q_proj CORRECT offset min={float(tq.min())} max={float(tq.max())} std={float(tq.float().std())}")
p("DONE")
