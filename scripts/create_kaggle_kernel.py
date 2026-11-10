import json
from pathlib import Path

kernel_dir = Path(r"d:\Coding\Hackathon\GFG\ARM\kaggle_staging\kernel")
kernel_dir.mkdir(parents=True, exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# RoadScribe 3B SLM Fine-Tuning (QLoRA)\n",
            "This notebook fine-tunes `Qwen/Qwen2.5-3B-Instruct` on the RoadScribe dataset for grounded driving telematics narratives.\n",
            "- **Architecture**: QLoRA 4-bit (NF4, rank 16, alpha 32)\n",
            "- **Target**: Capsule -> Plan -> Grounded Narrative\n",
            "- **Data**: 2,696 training pairs (including frontier Opus teacher examples)\n",
            "- **Evaluation**: Held-out E2 test personas + 9 real PVS trips\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Install & Verify Dependencies\n",
            "!pip install -q --upgrade transformers peft bitsandbytes accelerate datasets\n",
            "import torch\n",
            "print(f'CUDA available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'Device: {torch.cuda.get_device_name(0)}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Locate Dataset Files (Robust Search)\n",
            "import os, json, glob\n",
            "from pathlib import Path\n",
            "\n",
            "print('Listing /kaggle/input:')\n",
            "for p in Path('/kaggle/input').rglob('*'):\n",
            "    if p.is_file():\n",
            "        print(' ', p)\n",
            "\n",
            "# Search for train_sft.jsonl\n",
            "found = list(Path('/kaggle/input').rglob('train_sft.jsonl'))\n",
            "if found:\n",
            "    data_dir = found[0].parent\n",
            "    print(f'Found dataset at: {data_dir}')\n",
            "else:\n",
            "    print('Dataset not found in /kaggle/input, downloading via Kaggle API...')\n",
            "    os.system('kaggle datasets download -d nishkarshsinghal/roadscribe-dataset -p /kaggle/working/data --unzip')\n",
            "    data_dir = Path('/kaggle/working/data')\n",
            "\n",
            "train_path = data_dir / 'train_sft.jsonl'\n",
            "test_path = data_dir / 'test_e2.jsonl'\n",
            "\n",
            "with open(train_path, 'r', encoding='utf-8') as f:\n",
            "    sample = json.loads(f.readline())\n",
            "print('Sample persona:', sample.get('persona'))\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Format Data into Chat Conversations\n",
            "from datasets import Dataset\n",
            "\n",
            "def load_and_format(path):\n",
            "    records = []\n",
            "    with open(path, 'r', encoding='utf-8') as f:\n",
            "        for line in f:\n",
            "            item = json.loads(line)\n",
            "            persona = item.get('persona', 'commuter')\n",
            "            cap = item.get('capsule', '')\n",
            "            plan = item.get('plan', '')\n",
            "            narrative = item.get('narrative', '')\n",
            "            \n",
            "            if plan:\n",
            "                target = f\"<plan>\\n{json.dumps(plan) if isinstance(plan, dict) else str(plan)}\\n</plan>\\n\\n{narrative}\"\n",
            "            else:\n",
            "                target = narrative\n",
            "                \n",
            "            user_prompt = f\"Persona: {persona}\\n\\nTrip Capsule:\\n{cap}\\n\\nGenerate the driving trip summary adhering to the verified facts.\"\n",
            "            \n",
            "            records.append({\n",
            "                'messages': [\n",
            "                    {'role': 'system', 'content': 'You are RoadScribe, an edge vehicle telematics assistant. You summarize driving trips based on ISO 2631 vibration telemetry, road condition units, and detected events. Stick strictly to the capsule facts with zero hallucinations.'},\n",
            "                    {'role': 'user', 'content': user_prompt},\n",
            "                    {'role': 'assistant', 'content': target}\n",
            "                ]\n",
            "            })\n",
            "    return Dataset.from_list(records)\n",
            "\n",
            "train_ds = load_and_format(train_path)\n",
            "test_ds = load_and_format(test_path)\n",
            "print(f'Loaded {len(train_ds)} train examples and {len(test_ds)} test examples.')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Load Model in 4-bit NF4 Quantization\n",
            "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
            "\n",
            "model_id = 'Qwen/Qwen2.5-3B-Instruct'\n",
            "bnb_config = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_quant_type='nf4',\n",
            "    bnb_4bit_compute_dtype=torch.float16,\n",
            "    bnb_4bit_use_double_quant=True\n",
            ")\n",
            "\n",
            "tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n",
            "tokenizer.pad_token = tokenizer.eos_token\n",
            "\n",
            "model = AutoModelForCausalLM.from_pretrained(\n",
            "    model_id,\n",
            "    quantization_config=bnb_config,\n",
            "    device_map='auto',\n",
            "    trust_remote_code=True\n",
            ")\n",
            "print('Qwen-2.5-3B-Instruct loaded successfully in 4-bit!')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Format and Tokenize Prompts\n",
            "def tokenize_fn(example):\n",
            "    prompt_text = tokenizer.apply_chat_template(example['messages'], tokenize=False)\n",
            "    enc = tokenizer(prompt_text, max_length=1536, truncation=True)\n",
            "    enc['labels'] = enc['input_ids'].copy()\n",
            "    return enc\n",
            "\n",
            "train_tokenized = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)\n",
            "test_tokenized = test_ds.map(tokenize_fn, remove_columns=test_ds.column_names)\n",
            "print(f'Tokenized: train={len(train_tokenized)}, test={len(test_tokenized)}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Set up LoRA Config & Standard HF Trainer\n",
            "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
            "from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq\n",
            "\n",
            "model = prepare_model_for_kbit_training(model)\n",
            "\n",
            "peft_config = LoraConfig(\n",
            "    r=16,\n",
            "    lora_alpha=32,\n",
            "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],\n",
            "    lora_dropout=0.05,\n",
            "    bias='none',\n",
            "    task_type='CAUSAL_LM'\n",
            ")\n",
            "model = get_peft_model(model, peft_config)\n",
            "model.print_trainable_parameters()\n",
            "\n",
            "# Safe TrainingArguments compatible across all transformers versions (4.x and 5.x)\n",
            "training_args = TrainingArguments(\n",
            "    output_dir='/kaggle/working/roadscribe_output',\n",
            "    num_train_epochs=2,\n",
            "    per_device_train_batch_size=4,\n",
            "    gradient_accumulation_steps=4,\n",
            "    learning_rate=2e-4,\n",
            "    lr_scheduler_type='cosine',\n",
            "    warmup_steps=20,\n",
            "    logging_steps=25,\n",
            "    save_strategy='epoch',\n",
            "    fp16=True,\n",
            "    report_to='none'\n",
            ")\n",
            "\n",
            "data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)\n",
            "\n",
            "trainer = Trainer(\n",
            "    model=model,\n",
            "    train_dataset=train_tokenized,\n",
            "    args=training_args,\n",
            "    data_collator=data_collator,\n",
            ")\n",
            "\n",
            "print('Starting QLoRA training with standard HF Trainer...')\n",
            "train_res = trainer.train()\n",
            "print('Training complete!')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Save LoRA Adapter Weights\n",
            "save_dir = '/kaggle/working/roadscribe_3b_lora'\n",
            "model.save_pretrained(save_dir)\n",
            "tokenizer.save_pretrained(save_dir)\n",
            "print(f'Saved fine-tuned LoRA weights to: {save_dir}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8. Evaluate on Real PVS Trips 1..9\n",
            "model.eval()\n",
            "pvs_files = sorted(list(data_dir.glob('pvs_[1-9].json')))\n",
            "print(f'Found {len(pvs_files)} PVS real trip files.')\n",
            "\n",
            "pvs_results = []\n",
            "for pf in pvs_files:\n",
            "    with open(pf, 'r', encoding='utf-8') as f:\n",
            "        pdata = json.load(f)\n",
            "    \n",
            "    trip_id = pdata['trip_id']\n",
            "    capsule_text = pdata['capsule_text']\n",
            "    \n",
            "    for persona in ['commuter', 'sports_car']:\n",
            "        prompt = f\"Persona: {persona}\\n\\nTrip Capsule:\\n{capsule_text}\\n\\nGenerate the driving trip summary adhering to the verified facts.\"\n",
            "        messages = [\n",
            "            {'role': 'system', 'content': 'You are RoadScribe, an edge vehicle telematics assistant. You summarize driving trips based on ISO 2631 vibration telemetry, road condition units, and detected events. Stick strictly to the capsule facts with zero hallucinations.'},\n",
            "            {'role': 'user', 'content': prompt}\n",
            "        ]\n",
            "        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
            "        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)\n",
            "        \n",
            "        with torch.no_grad():\n",
            "            outputs = model.generate(\n",
            "                **inputs,\n",
            "                max_new_tokens=300,\n",
            "                temperature=0.3,\n",
            "                do_sample=True,\n",
            "                top_p=0.9,\n",
            "                pad_token_id=tokenizer.eos_token_id\n",
            "            )\n",
            "        \n",
            "        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)\n",
            "        pvs_results.append({\n",
            "            'trip_id': trip_id,\n",
            "            'persona': persona,\n",
            "            'generated_narrative': gen_text\n",
            "        })\n",
            "        print(f'Trip {trip_id} ({persona}) generated ({len(gen_text.split())} words)')\n",
            "\n",
            "# Save results\n",
            "with open('/kaggle/working/pvs_predictions.json', 'w', encoding='utf-8') as f:\n",
            "    json.dump(pvs_results, f, indent=2)\n",
            "print('PVS generation complete! All outputs saved to /kaggle/working/pvs_predictions.json')\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = kernel_dir / "roadscribe_3b_qlora.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

kmeta = {
    "id": "nishkarshsinghal/roadscribe-3b-qlora-trainer",
    "title": "RoadScribe 3B QLoRA Trainer",
    "code_file": "roadscribe_3b_qlora.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["nishkarshsinghal/roadscribe-dataset"]
}

with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(kmeta, f, indent=2)

print(f"Kernel updated at {nb_path}")
