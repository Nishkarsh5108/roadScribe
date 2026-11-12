"""P5: RoadScribe narrator trained on its intended PLAN and REALIZE tasks, then run on PVS (2026-11-12).

1. QLoRA fine-tune of Qwen2.5-3B-Instruct (r 16, alpha 32, lr 2e-4, 1 epoch), loss on assistant tokens only.
2. For each PVS prompt (capsule + fact index) and seed: generate a plan, validate it against the ledger,
   ground it deterministically into fact frames, and let the model realize the frames into text.
Outputs: /kaggle/working/pgr_v2_results.json, the adapter in /kaggle/working/pgr_v2_lora.
"""
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "peft", "bitsandbytes", "accelerate"], check=False)

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,  # noqa: E402
                          Trainer, TrainingArguments)

INP, OUT = Path("/kaggle/input"), Path("/kaggle/working")
root = [p for p in INP.rglob("train.jsonl")][0].parent
src = [p.parent.parent for p in INP.rglob("roadscribe/__init__.py")][0]
sys.path.insert(0, str(src))
from roadscribe.layer3.plan_grammar import validate_plan  # noqa: E402
from roadscribe.layer3.templates import ground  # noqa: E402
from roadscribe.synth.formats import SYSTEM_REALIZE, persona_card  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_LEN = 2048
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token


def encode(e):
    msgs = e["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(f_ids) > MAX_LEN or f_ids[:len(p_ids)] != p_ids:
        return None
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]}


raw = [json.loads(l) for l in open(root / "train.jsonl", encoding="utf-8")]
data = [x for x in map(encode, raw) if x is not None]
print(f"training examples kept {len(data)} of {len(raw)}; mean length {sum(len(d['input_ids']) for d in data) / len(data):.0f}", flush=True)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, xs):
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i]


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
args = TrainingArguments(output_dir=str(OUT / "ckpt"), num_train_epochs=1, per_device_train_batch_size=1,
                         gradient_accumulation_steps=16, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10,
                         logging_steps=10, save_strategy="no", fp16=True, gradient_checkpointing=True,
                         gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=ListDataset(data),
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
res = trainer.train()
print(f"training done in {time.time() - t0:.0f}s; loss {res.training_loss:.4f}", flush=True)
model.save_pretrained(OUT / "pgr_v2_lora")
tok.save_pretrained(OUT / "pgr_v2_lora")
json.dump(trainer.state.log_history, open(OUT / "train_log.json", "w"))

# ---------------------------------------------------------------- generation on PVS
model.eval()
model.gradient_checkpointing_disable()
model.config.use_cache = True
tok.padding_side = "left"


def generate(batch_msgs, seed, max_new):
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, temperature=0.3, do_sample=True, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    res = []
    for row in out:
        new = row[enc["input_ids"].shape[1]:]
        res.append((tok.decode(new, skip_special_tokens=True), int((new != tok.eos_token_id).sum())))
    return res


def parse_plan(text):
    t = text.strip()
    if "<plan>" in t:
        t = t.split("<plan>")[1].split("</plan>")[0]
    t = t.strip().strip("`")
    if t.startswith("json"):
        t = t[4:]
    a, b = t.find("{"), t.rfind("}")
    return json.loads(t[a:b + 1])


prompts = json.loads((root / "pvs_prompts.json").read_text(encoding="utf-8"))
ledgers = pickle.load(open(root / "pvs_ledgers.pkl", "rb"))
results = []
for seed in (0, 1, 2):
    for i in range(0, len(prompts), 6):
        chunk = prompts[i:i + 6]
        plans = generate([p["messages"] for p in chunk], seed * 1000 + i, 768)
        realize_jobs = []
        for p, (ptext, n_tok) in zip(chunk, plans):
            r = dict(key=p["key"], condition=p["condition"], trip_id=p["trip_id"], persona=p["persona"], seed=seed,
                     plan_text=ptext, plan_tokens=n_tok, plan_parses=False, plan_errors=None, frames=None, narrative=None)
            L = ledgers[p["key"]]
            try:
                plan = parse_plan(ptext)
                r["plan_parses"] = True
                errs = validate_plan(plan, L["ledger"], L["visible"])
                r["plan_errors"] = errs[:5]
                r["n_plan_errors"] = len(errs)
                if not errs:
                    frames = ground(plan, L["ledger"])
                    r["frames"] = frames
                    msgs = [{"role": "system", "content": SYSTEM_REALIZE},
                            {"role": "user", "content": f"{persona_card(p['persona'])}\n\nFRAMES\n"
                                                        f"{json.dumps(frames, separators=(',', ':'), sort_keys=True)}"}]
                    realize_jobs.append((r, msgs))
            except Exception as ex:  # noqa: BLE001
                r["plan_errors"] = [f"{type(ex).__name__}: {ex}"[:200]]
            results.append(r)
        if realize_jobs:
            outs = generate([m for _, m in realize_jobs], seed * 1000 + i + 1, 768)
            for (r, _), (txt, n_tok) in zip(realize_jobs, outs):
                r["narrative"], r["narrative_tokens"] = txt, n_tok
        print(f"seed {seed} batch {i // 6}: {sum(r['frames'] is not None for r in results)} valid plans so far "
              f"({time.time() - t0:.0f}s)", flush=True)
        (OUT / "pgr_v2_results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")

print("finished:", len(results), "plans;", sum(r["frames"] is not None for r in results), "valid", flush=True)
