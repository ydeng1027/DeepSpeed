from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "mistralai/Mixtral-8x7B-v0.1"
save_path = f"fp16_cpu/{model_id}"
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)