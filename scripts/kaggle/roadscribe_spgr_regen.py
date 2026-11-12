"""RoadScribe S-PGR regeneration on PVS capsules (2026-11-12).

Same base model, LoRA adapter, system prompt, user prompt and sampling as the 2026-11-10 run
(temperature 0.3, top_p 0.9). The only intended change is the generation budget:
max_new_tokens 1024 instead of 300, because the plan used most of the 300 tokens and the
narratives were cut off. Three seeds per (capsule, persona), for the original capsules and for
capsules computed with per-vehicle calibration on shared roads (identifiability study P4).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "peft", "bitsandbytes", "accelerate"], check=False)

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

INP = Path("/kaggle/input")
OUT = Path("/kaggle/working")
for p in sorted(INP.rglob("*")):
    print("input:", p)
adapter_dir = [p.parent for p in INP.rglob("adapter_config.json")][0]
_all_caps = sorted(INP.rglob("pvs_*.json"))
caps = {"orig": [p for p in _all_caps if any("caps_orig" in part for part in p.parts)],
        "cal": [p for p in _all_caps if any("caps_cal" in part for part in p.parts)]}
assert len(caps["orig"]) == 9, f"expected 9 original capsules, found {len(caps['orig'])}"
print("adapter:", adapter_dir, {k: len(v) for k, v in caps.items()}, flush=True)

SYSTEM = ("You are RoadScribe, an edge vehicle telematics assistant. You summarize driving trips based on ISO 2631 "
          "vibration telemetry, road condition units, and detected events. Stick strictly to the capsule facts with "
          "zero hallucinations.")
PERSONAS = ("commuter", "sports_car")
SEEDS = (0, 1, 2)
MAX_NEW = 1024

tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", quantization_config=bnb, device_map="auto",
                                            trust_remote_code=True)
model = PeftModel.from_pretrained(base, str(adapter_dir))
model.eval()
print("model loaded on", torch.cuda.get_device_name(0), flush=True)


def build(persona, capsule_text):
    user = (f"Persona: {persona}\n\nTrip Capsule:\n{capsule_text}\n\n"
            "Generate the driving trip summary adhering to the verified facts.")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


results = []
t_start = time.time()
for cond, paths in caps.items():
    items = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        for persona in PERSONAS:
            items.append((d["trip_id"], persona, build(persona, d["capsule_text"])))
    for seed in SEEDS:
        for i in range(0, len(items), 6):
            chunk = items[i:i + 6]
            enc = tok([c[2] for c in chunk], return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(seed * 1000 + i)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, temperature=0.3, do_sample=True, top_p=0.9,
                                     pad_token_id=tok.eos_token_id)
            for (trip_id, persona, _), row in zip(chunk, out):
                new = row[enc["input_ids"].shape[1]:]
                n_new = int((new != tok.eos_token_id).sum())
                text = tok.decode(new, skip_special_tokens=True)
                results.append(dict(condition=cond, trip_id=trip_id, persona=persona, seed=seed, generated_narrative=text,
                                    n_new_tokens=n_new, hit_budget=n_new >= MAX_NEW))
            print(f"{cond} seed {seed} batch {i // 6}: done ({time.time() - t_start:.0f}s)", flush=True)
            (OUT / "spgr_regen.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

(OUT / "spgr_regen.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("finished", len(results), "generations;", sum(r["hit_budget"] for r in results), "hit the budget", flush=True)
