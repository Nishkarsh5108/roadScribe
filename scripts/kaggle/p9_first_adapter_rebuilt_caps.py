"""P9 (exploratory, declared in WORKLOG before running): the FIRST adapter (2026-11-10) on the REBUILT PVS capsules
(no spurious map channel, persona fixed), as used by P5. Isolates the capsule fix from the plan-ground-realize
retraining. Same system prompt, user prompt, sampling (T 0.3, top-p 0.9) and budget (1,024) as the regeneration.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "peft", "bitsandbytes", "accelerate"], check=False)

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

INP, OUT = Path("/kaggle/input"), Path("/kaggle/working")
adapters = [p.parent for p in INP.rglob("adapter_config.json")]
assert len(adapters) == 1, adapters  # only roadscribe-regen-inputs carries an adapter (the 2026-11-10 one)
adapter_dir = adapters[0]
caps = json.loads('[{"key": "orig|pvs1|commuter", "trip_id": "pvs1", "persona": "commuter", "capsule": "TRIP pvs1 | 24.0 min | 13.8 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU1 0.0-4.3km - | felt 2 SHOCK | bump x12; jolt x2\\nU2 4.3-9.4km - | felt 4 SHOCK | bump x7; jolt x1\\nU3 9.4-13.8km - | felt 2 SHOCK | artifact x3; bump x8"}, {"key": "orig|pvs1|sports_car", "trip_id": "pvs1", "persona": "sports_car", "capsule": "TRIP pvs1 | 24.0 min | 13.8 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU1 0.0-4.3km - | felt 2 SHOCK | bump x12; jolt x2\\nU2 4.3-9.4km - | felt 4 SHOCK | bump x7; jolt x1\\nU3 9.4-13.8km - | felt 2 SHOCK | artifact x3; bump x8"}, {"key": "orig|pvs2|commuter", "trip_id": "pvs2", "persona": "commuter", "capsule": "TRIP pvs2 | 20.8 min | 11.5 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU1 0.0-0.4km - | felt 1 SHOCK | jolt x1; bump x7\\nU2 0.4-2.5km - | felt 4 SHOCK | bump x8\\nU3 2.5-5.8km - | felt 2 SHOCK | bump x14; jolt x7; artifact x3\\nU4 5.8-7.2km - | felt 4 SHOCK | bump x4\\nU5 7.2-11.5km - | felt 2 SHOCK | jolt x3; bump x4"}, {"key": "orig|pvs2|sports_car", "trip_id": "pvs2", "persona": "sports_car", "capsule": "TRIP pvs2 | 20.8 min | 11.5 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU1 0.0-0.4km - | felt 1 SHOCK | jolt x1; bump x7\\nU2 0.4-2.5km - | felt 4 SHOCK | bump x8\\nU3 2.5-5.8km - | felt 2 SHOCK | bump x14; jolt x7; artifact x3\\nU4 5.8-7.2km - | felt 4 SHOCK | bump x4\\nU5 7.2-11.5km - | felt 2 SHOCK | jolt x3; bump x4"}, {"key": "orig|pvs3|commuter", "trip_id": "pvs3", "persona": "commuter", "capsule": "TRIP pvs3 | 17.6 min | 10.7 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU1 0.0-3.0km - | felt 2 SHOCK | bump x8\\nU2 3.0-6.3km - | felt 4 SHOCK | bump x4; jolt x1\\nU3 6.3-10.7km - | felt 2 SHOCK | jolt x2; bump x6"}, {"key": "orig|pvs3|sports_car", "trip_id": "pvs3", "persona": "sports_car", "capsule": "TRIP pvs3 | 17.6 min | 10.7 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU1 0.0-3.0km - | felt 2 SHOCK | bump x8\\nU2 3.0-6.3km - | felt 4 SHOCK | bump x4; jolt x1\\nU3 6.3-10.7km - | felt 2 SHOCK | jolt x2; bump x6"}, {"key": "orig|pvs4|commuter", "trip_id": "pvs4", "persona": "commuter", "capsule": "TRIP pvs4 | 22.1 min | 13.8 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU1 0.0-4.3km - | felt 1 SHOCK | bump x14; jolt x4\\nU2 4.3-9.4km - | felt 3 SHOCK | bump x6; jolt x1\\nU3 9.4-13.8km - | felt 1 SHOCK | bump x7; jolt x1"}, {"key": "orig|pvs4|sports_car", "trip_id": "pvs4", "persona": "sports_car", "capsule": "TRIP pvs4 | 22.1 min | 13.8 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU1 0.0-4.3km - | felt 1 SHOCK | bump x14; jolt x4\\nU2 4.3-9.4km - | felt 3 SHOCK | bump x6; jolt x1\\nU3 9.4-13.8km - | felt 1 SHOCK | bump x7; jolt x1"}, {"key": "orig|pvs5|commuter", "trip_id": "pvs5", "persona": "commuter", "capsule": "TRIP pvs5 | 22.3 min | 11.5 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU2 0.4-2.5km - | felt 3 SHOCK | bump x14\\nU3 2.5-5.8km - | felt 1 SHOCK | bump x19; jolt x9; artifact x2\\nU4 5.8-7.2km - | felt 3 SHOCK | bump x1\\nU5 7.2-11.5km - | felt 1 SHOCK | jolt x2; bump x7\\nREST 1 unit, 0.4 km, surface -, felt 1-1, 7 events"}, {"key": "orig|pvs5|sports_car", "trip_id": "pvs5", "persona": "sports_car", "capsule": "TRIP pvs5 | 22.3 min | 11.5 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU2 0.4-2.5km - | felt 3 SHOCK | bump x14\\nU3 2.5-5.8km - | felt 1 SHOCK | bump x19; jolt x9; artifact x2\\nU4 5.8-7.2km - | felt 3 SHOCK | bump x1\\nU5 7.2-11.5km - | felt 1 SHOCK | jolt x2; bump x7\\nREST 1 unit, 0.4 km, surface -, felt 1-1, 7 events"}, {"key": "orig|pvs6|commuter", "trip_id": "pvs6", "persona": "commuter", "capsule": "TRIP pvs6 | 16.0 min | 10.7 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU1 0.0-3.0km - | felt 2 SHOCK | bump x12; jolt x1\\nU2 3.0-6.3km - | felt 3 SHOCK | bump x4\\nU3 6.3-10.7km - | felt 1 SHOCK | jolt x2; bump x10; artifact x1"}, {"key": "orig|pvs6|sports_car", "trip_id": "pvs6", "persona": "sports_car", "capsule": "TRIP pvs6 | 16.0 min | 10.7 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU1 0.0-3.0km - | felt 2 SHOCK | bump x12; jolt x1\\nU2 3.0-6.3km - | felt 3 SHOCK | bump x4\\nU3 6.3-10.7km - | felt 1 SHOCK | jolt x2; bump x10; artifact x1"}, {"key": "orig|pvs7|commuter", "trip_id": "pvs7", "persona": "commuter", "capsule": "TRIP pvs7 | 21.4 min | 13.8 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU1 0.0-4.3km - | felt 2 SHOCK | bump x10; jolt x2\\nU2 4.3-9.4km - | felt 4 SHOCK | bump x20\\nU3 9.4-13.8km - | felt 2 SHOCK | jolt x1; bump x5"}, {"key": "orig|pvs7|sports_car", "trip_id": "pvs7", "persona": "sports_car", "capsule": "TRIP pvs7 | 21.4 min | 13.8 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU1 0.0-4.3km - | felt 2 SHOCK | bump x10; jolt x2\\nU2 4.3-9.4km - | felt 4 SHOCK | bump x20\\nU3 9.4-13.8km - | felt 2 SHOCK | jolt x1; bump x5"}, {"key": "orig|pvs8|commuter", "trip_id": "pvs8", "persona": "commuter", "capsule": "TRIP pvs8 | 20.6 min | 11.5 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU1 0.0-0.4km - | felt 1 SHOCK | bump x14\\nU2 0.4-2.5km - | felt 4 SHOCK | bump x13\\nU3 2.5-5.8km - | felt 2 SHOCK | bump x19; jolt x3\\nU4 5.8-7.2km - | felt 4 SHOCK | bump x2\\nU5 7.2-11.5km - | felt 2 SHOCK | jolt x2; bump x7"}, {"key": "orig|pvs8|sports_car", "trip_id": "pvs8", "persona": "sports_car", "capsule": "TRIP pvs8 | 20.6 min | 11.5 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU1 0.0-0.4km - | felt 1 SHOCK | bump x14\\nU2 0.4-2.5km - | felt 4 SHOCK | bump x13\\nU3 2.5-5.8km - | felt 2 SHOCK | bump x19; jolt x3\\nU4 5.8-7.2km - | felt 4 SHOCK | bump x2\\nU5 7.2-11.5km - | felt 2 SHOCK | jolt x2; bump x7"}, {"key": "orig|pvs9|commuter", "trip_id": "pvs9", "persona": "commuter", "capsule": "TRIP pvs9 | 15.3 min | 10.7 km | persona=commuter | ch=acc,gyro,gps | hist=none\\nU1 0.0-3.0km - | felt 2 SHOCK | bump x10; jolt x2\\nU2 3.0-6.3km - | felt 4 SHOCK | jolt x1; bump x10\\nU3 6.3-10.7km - | felt 2 SHOCK | jolt x2; bump x5"}, {"key": "orig|pvs9|sports_car", "trip_id": "pvs9", "persona": "sports_car", "capsule": "TRIP pvs9 | 15.3 min | 10.7 km | persona=sports_car | ch=acc,gyro,gps | hist=none\\nU1 0.0-3.0km - | felt 2 SHOCK | bump x10; jolt x2\\nU2 3.0-6.3km - | felt 4 SHOCK | jolt x1; bump x10\\nU3 6.3-10.7km - | felt 2 SHOCK | jolt x2; bump x5"}]')  # the 18 rebuilt capsules, embedded
print("adapter:", adapter_dir, "capsules:", len(caps), flush=True)
SYSTEM = ("You are RoadScribe, an edge vehicle telematics assistant. You summarize driving trips based on ISO 2631 "
          "vibration telemetry, road condition units, and detected events. Stick strictly to the capsule facts with "
          "zero hallucinations.")
tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, str(adapter_dir))
model.eval()


def build(persona, capsule):
    user = f"Persona: {persona}\n\nTrip Capsule:\n{capsule}\n\nGenerate the driving trip summary adhering to the verified facts."
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


results, t0 = [], time.time()
for seed in (0, 1, 2):
    for i in range(0, len(caps), 6):
        chunk = caps[i:i + 6]
        enc = tok([build(c["persona"], c["capsule"]) for c in chunk], return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed * 1000 + i)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1024, temperature=0.3, do_sample=True, top_p=0.9, pad_token_id=tok.eos_token_id)
        for c, row in zip(chunk, out):
            new = row[enc["input_ids"].shape[1]:]
            results.append(dict(key=c["key"], trip_id=c["trip_id"], persona=c["persona"], seed=seed,
                                generated_narrative=tok.decode(new, skip_special_tokens=True), n_new_tokens=int((new != tok.eos_token_id).sum())))
        (OUT / "p9_first_adapter_rebuilt_caps.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"seed {seed} batch {i // 6} ({time.time() - t0:.0f}s)", flush=True)
print("finished", len(results), flush=True)
