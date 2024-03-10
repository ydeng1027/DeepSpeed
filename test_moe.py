import mii
pipe = mii.pipeline("NousResearch/Llama-2-70b-hf", quantization_mode='wf6af16')
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)