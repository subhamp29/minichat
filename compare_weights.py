import json, struct, os, torch, types
import train_lora as T  # placeholder

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
SF = os.path.join(SNAP, "model.safetensors")

# Parse header to get byte offsets/dtype for specific tensors.
with open(SF, "rb") as f:
    hl = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hl).decode("utf-8"))

DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool}
ESZ = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1, "BOOL": 1}

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))

names = ["model.embed_tokens.weight",
         "model.layers.0.mlp.gate_proj.weight",
         "model.layers.0.self_attn.q_proj.weight",
         "model.layers.12.mlp.up_proj.weight",
         "model.norm.weight"]

with open(SF, "rb") as f:
    for nm in names:
        info = header[nm]
        start, end = info["data_offsets"]
        f.seek(start)
        raw = f.read(end - start)  # buffered read, no mmap
        ref = torch.frombuffer(raw, dtype=DT[info["dtype"]], count=len(raw) // ESZ[info["dtype"]])
        ref = ref.reshape(info["shape"])
        mod_name, leaf = nm.rsplit(".", 1)
        mine = getattr(m.get_submodule(mod_name), leaf)
        ok = torch.allclose(ref, mine)
        diff = float((ref - mine).abs().max()) if not ok else 0.0
        # also report scale of the tensor
        print(f"{nm}: dtype={info['dtype']} shape={tuple(info['shape'])} "
              f"match={bool(ok)} max_abs_diff={diff:.3e} "
              f"mine_max={float(mine.float().abs().max()):.3e}", flush=True)
print("DONE", flush=True)
