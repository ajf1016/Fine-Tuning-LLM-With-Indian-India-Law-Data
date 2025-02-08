import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

# ✅ Use Mistral 7B
model_id = "mistralai/Mistral-7B-v0.1"

# ✅ Enable QLoRA (4-bit Quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # ✅ Ensure PyTorch dtype is set
    bnb_4bit_quant_type="nf4",  # ✅ Use `nf4` for Mac compatibility
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoGPTQForCausalLM.from_pretrained(model_id, device_map="auto")

print("✅ Mistral 7B (4-bit) Loaded Successfully!")

# ✅ Load Dataset
dataset = load_dataset("json", data_files={
                       "train": "legal_finetune_data.json"})
tokenizer.pad_token = tokenizer.eos_token

# ✅ Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

print("✅ LoRA Applied Successfully!")

# ✅ Training Settings
training_args = TrainingArguments(
    output_dir="./mistral_legal_finetune",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="epoch",
    evaluation_strategy="no",
    logging_dir="./logs"
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

# ✅ Start Training
trainer.train()
