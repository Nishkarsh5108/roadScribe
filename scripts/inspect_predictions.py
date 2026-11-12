import os
import sys
import json
import re

os.environ['ROADSCRIBE_PVS_DIR'] = r'd:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle'
sys.path.insert(0, r'd:\Coding\Hackathon\GFG\ARM\roadScribe')

with open(r'd:\Coding\Hackathon\GFG\ARM\roadScribe\data\models\kaggle_run\pvs_predictions.json', 'r', encoding='utf-8') as f:
    preds = json.load(f)

print(f"Total predictions loaded: {len(preds)}")
valid_plans = 0

for i, p in enumerate(preds):
    raw_text = p['generated_narrative']
    has_plan = '<plan>' in raw_text and '</plan>' in raw_text
    plan_json = None
    narrative = raw_text
    if has_plan:
        plan_part = raw_text.split('<plan>')[1].split('</plan>')[0].strip()
        narrative = raw_text.split('</plan>')[1].strip()
        try:
            plan_json = json.loads(plan_part)
            valid_plans += 1
        except Exception as e:
            plan_json = f'INVALID JSON: {e}'
    
    trip_id = p.get('trip_id')
    persona = p.get('persona')
    n_words = len(narrative.split())
    is_valid = isinstance(plan_json, dict)
    print(f"[{i+1:02d}/18] Trip {trip_id:4s} | Persona: {persona:12s} | Plan Valid: {str(is_valid):5s} | Words: {n_words:3d}")
    print(f"     Narrative: {narrative[:120]}...")

print(f"\nSummary: {valid_plans}/{len(preds)} valid JSON plans ({valid_plans/len(preds)*100:.1f}%)")

