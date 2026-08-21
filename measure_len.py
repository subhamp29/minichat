from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

lens = []
maxlen = 0
with open("finetune_dataset.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        text = tok.apply_chat_template(o["messages"], tokenize=False, add_generation_prompt=True)
        if not text.endswith(tok.eos_token):
            text = text + tok.eos_token
        ids = tok(text, add_special_tokens=False)["input_ids"]
        lens.append(len(ids))
        maxlen = max(maxlen, len(ids))

lens.sort()
print("num examples:", len(lens))
print("max tokens:", maxlen)
print("median:", lens[len(lens)//2])
print("90th:", lens[int(len(lens)*0.9)])
print("recommended max_seq_len:", max(512, ((maxlen + 63)//64)*64))  # round up to 64
