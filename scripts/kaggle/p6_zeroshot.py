"""P6 companion: prompted (not fine-tuned) Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct on arm B's input
(capsule + grounded frames) with the repository's realize instruction. 54 narratives per model.
"""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "bitsandbytes", "accelerate"], check=False)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

INP, OUT = Path("/kaggle/input"), Path("/kaggle/working")
root = [p for p in INP.rglob("pvs_zeroshot.json")][0].parent
prompts = json.loads((root / "pvs_zeroshot.json").read_text(encoding="utf-8"))
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
t0 = time.time()
results = []
for model_id in ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.eval()
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
                results.append(dict(model=model_id, key=p["key"], trip_id=p["trip_id"], persona=p["persona"], seed=seed,
                                    narrative=tok.decode(new, skip_special_tokens=True), n_new_tokens=int((new != tok.eos_token_id).sum())))
            (OUT / "p6_zeroshot.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(f"{model_id} seed {seed} batch {i // 6} ({time.time() - t0:.0f}s)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
print("finished", len(results), flush=True)
