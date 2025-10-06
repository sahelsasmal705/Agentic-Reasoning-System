from peft import PeftModel # type: ignore

from transformers import AutoModelForCausalLM

# Load your base model (replace 'your-model-name' with the actual model name)
base_model = AutoModelForCausalLM.from_pretrained("your-model-name")

# Wrap the base model with PEFT (replace with actual PEFT config if needed)
peft_model = PeftModel(base_model) # type: ignore

peft_model.save_pretrained("outputs/lora-gptj")