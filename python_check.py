import mii
pipe = mii.pipeline("/home/chengli/Downloads/fp16_cpu/mistralai/Mixtral-8x7B-v0.1", quantization_mode='wf6af16')
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)
