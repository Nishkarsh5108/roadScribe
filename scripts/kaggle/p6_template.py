"""P6 arm __ARM__ (PREREG_P6.md): QLoRA fine-tune on train___ARM__.jsonl, then 54 PVS narratives.

Identical across arms except the input content (A capsule; B + frames; C + frames without positions).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "peft", "bitsandbytes", "accelerate"], check=False)

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,  # noqa: E402
                          Trainer, TrainingArguments)

ARM = "__ARM__"
INP, OUT = Path("/kaggle/input"), Path("/kaggle/working")
root = [p for p in INP.rglob(f"train_{ARM}.jsonl")][0].parent
MODEL, MAX_LEN = "Qwen/Qwen2.5-3B-Instruct", 3072
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token


def encode(e):
    msgs = e["messages"]
    p = tok(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    f = tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"]
    if len(f) > MAX_LEN or f[:len(p)] != p:
        return None
    return {"input_ids": f, "attention_mask": [1] * len(f), "labels": [-100] * len(p) + f[len(p):]}


raw = [json.loads(l) for l in open(root / f"train_{ARM}.jsonl", encoding="utf-8")]
data = [x for x in map(encode, raw) if x is not None]
print(f"arm {ARM}: kept {len(data)} of {len(raw)}; mean tokens {sum(len(d['input_ids']) for d in data) / len(data):.0f}", flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, xs):
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i]


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
args = TrainingArguments(output_dir=str(OUT / "ckpt"), num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=16,
                         learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=5, logging_steps=10, save_strategy="no", fp16=True,
                         gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none", seed=20261113)
trainer = Trainer(model=model, args=args, train_dataset=DS(data), data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
res = trainer.train()
print(f"arm {ARM}: training {time.time() - t0:.0f}s loss {res.training_loss:.4f}", flush=True)
json.dump(trainer.state.log_history, open(OUT / f"train_log_{ARM}.json", "w"))
model.save_pretrained(OUT / f"lora_{ARM}")

model.eval()
model.gradient_checkpointing_disable()
model.config.use_cache = True
tok.padding_side = "left"
prompts = json.loads((root / f"pvs_{ARM}.json").read_text(encoding="utf-8"))
results = []
for seed in (0, 1, 2):
    for i in range(0, len(prompts), 6):
        chunk = prompts[i:i + 6]
        texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed * 1000 + i)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=768, temperature=0.3, do_sample=True, top_p=0.9, pad_token_id=tok.eos_token_id)
        for p, row in zip(chunk, out):
            new = row[enc["input_ids"].shape[1]:]
            results.append(dict(arm=ARM, key=p["key"], trip_id=p["trip_id"], persona=p["persona"], seed=seed,
                                narrative=tok.decode(new, skip_special_tokens=True), n_new_tokens=int((new != tok.eos_token_id).sum())))
        (OUT / f"p6_{ARM}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"arm {ARM}: seed {seed} batch {i // 6} ({time.time() - t0:.0f}s)", flush=True)
print(f"arm {ARM}: finished {len(results)}", flush=True)
