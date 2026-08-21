import faulthandler, types, torch
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=False)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)
import train_lora as T

args = types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False)
m = T.load_base_model_cpu(args)
print("model loaded; entering train/forward+backward test", flush=True)
m.train()
ids = torch.randint(0, 151936, (1, 16))
print("forward with grad...", flush=True)
out = m(input_ids=ids)
print("forward done; loss=", float(out.logits.sum()), flush=True)
print("backward...", flush=True)
out.logits.sum().backward()
print("backward done; ok", flush=True)
