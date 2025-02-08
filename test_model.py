from transformers import pipeline

pipe = pipeline("text-generation", model="./mistral_legal_finetune",
                tokenizer="mistralai/Mistral-7B-v0.1")

query = "Generate an FIR for domestic violence under IPC 498A."
response = pipe(query, max_length=500)

print("\n🔹 Generated FIR:\n", response[0]["generated_text"])
