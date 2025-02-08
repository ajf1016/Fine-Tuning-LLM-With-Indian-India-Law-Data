import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

# Define model paths
base_model_id = "Qwen/Qwen1.5-0.5B"  # ✅ Load the base model from Hugging Face
adapter_path = "/Users/ajmalfayiz1016/Desktop/fine_tune_llama_on_indian_law/qwen_fine_tuned"

# Load base model
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned LoRA adapter & merge with base model
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # ✅ Merge LoRA adapter into the base model

# Move to GPU/CPU
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():  # Check for Apple Silicon GPU support
    device = torch.device("mps")
    print("✅ Using Apple MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("⚠️ GPU not available, using CPU")
model.to(device)

print("✅ Model loaded successfully with fine-tuned weights!")


def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=max_length,
                            temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Test with a legal prompt
prompt = "What is the ipc section of murder"
response = generate_response(prompt)
print("\nGenerated Response:\n", response)
