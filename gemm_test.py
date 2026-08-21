import torch
torch.set_num_threads(4)
print("torch", torch.__version__, flush=True)

# 1) tiny frombuffer on bytearray
ba = bytearray(20)
t = torch.frombuffer(ba, dtype=torch.bfloat16, count=10)
print("tiny frombuffer bytearray ok:", t.shape, t.dtype, flush=True)

# 2) random GEMM at exact lm_head shapes, bf16 vs fp32
a = torch.randn(2, 896, dtype=torch.bfloat16)
for N, dt, lbl in [(151936, torch.bfloat16, "vocab-bf16"),
                   (4864, torch.bfloat16, "mlp-bf16"),
                   (151936, torch.float32, "vocab-fp32")]:
    b = torch.randn(896, N, dtype=dt)
    with torch.no_grad():
        c = a.to(dt) @ b
    print(f"{lbl}: out max={float(c.max()):.4f} min={float(c.min()):.4f} std={float(c.std()):.4f}", flush=True)
print("DONE", flush=True)
