"""P8 (exploratory follow-up to P6, predictions in WORKLOG before running): is arm B's mis-copying a matter
of decoding or of training length?  MODE greedy: arm B's adapter, greedy decoding, one run per PVS input (18).
MODE epoch2: arm B's adapter trained one more epoch on train_B (same settings), then the P6 sampling (54).
"""
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "peft", "bitsandbytes", "accelerate"], check=False)

import torch  # noqa: E402
from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,  # noqa: E402
                          Trainer, TrainingArguments)

MODE = "__MODE__"
INP, OUT = Path("/kaggle/input"), Path("/kaggle/working")
root = [p for p in INP.rglob("train_B.jsonl")][0].parent
lora = [p for p in INP.rglob("adapter_config.json")][0].parent
MODEL, MAX_LEN = "Qwen/Qwen2.5-3B-Instruct", 3072
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)

if MODE == "epoch2":
    base = prepare_model_for_kbit_training(base)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, str(lora), is_trainable=True)

    def encode(e):
        msgs = e["messages"]
        p = tok(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        f = tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"]
        if len(f) > MAX_LEN or f[:len(p)] != p:
            return None
        return {"input_ids": f, "attention_mask": [1] * len(f), "labels": [-100] * len(p) + f[len(p):]}

    data = [x for x in map(encode, (json.loads(l) for l in open(root / "train_B.jsonl", encoding="utf-8"))) if x is not None]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            return data[i]

    args = TrainingArguments(output_dir=str(OUT / "ckpt"), num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=16,
                             learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=5, logging_steps=10, save_strategy="no", fp16=True,
                             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none", seed=20261113)
    res = Trainer(model=model, args=args, train_dataset=DS(), data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    print(f"epoch2: kept {len(data)}; training {time.time() - t0:.0f}s loss {res.training_loss:.4f}", flush=True)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
else:
    model = PeftModel.from_pretrained(base, str(lora))
model.eval()
tok.padding_side = "left"
prompts = json.loads((root / "pvs_B.json").read_text(encoding="utf-8"))
results = []
seeds = (0,) if MODE == "greedy" else (0, 1, 2)
BS = 3
for seed in seeds:
    for i in range(0, len(prompts), BS):
        chunk = prompts[i:i + BS]
        texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed * 1000 + i)
        kw = dict(do_sample=False) if MODE == "greedy" else dict(do_sample=True, temperature=0.3, top_p=0.9)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=768, pad_token_id=tok.eos_token_id, **kw)
        for p, row in zip(chunk, out):
            new = row[enc["input_ids"].shape[1]:]
            results.append(dict(arm=f"B-{MODE}", key=p["key"], trip_id=p["trip_id"], persona=p["persona"], seed=seed,
                                narrative=tok.decode(new, skip_special_tokens=True), n_new_tokens=int((new != tok.eos_token_id).sum())))
        (OUT / f"p8_{MODE}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"{MODE}: seed {seed} batch {i // BS} ({time.time() - t0:.0f}s)", flush=True)
print(f"{MODE}: finished {len(results)}", flush=True)
