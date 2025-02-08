import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

# ✅ Best alternative to Llama 2 (Legal fine-tuning)
model_id = "meta-llama/Meta-Llama-3-8B"

# ✅ Fix tokenizer error (Install sentencepiece if missing)
os.system("pip install sentencepiece protobuf")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# ✅ Load your Indian Legal Dataset
dataset = load_dataset("json", data_files={
                       "train": "legal_finetune_data.json"})

# ✅ Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./mistral_legal_finetune",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="epoch",
    evaluation_strategy="no",
    logging_dir="./logs"
)

# ✅ Data Collator (Ensures Padding)
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# ✅ Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ Start Fine-Tuning
trainer.train()
