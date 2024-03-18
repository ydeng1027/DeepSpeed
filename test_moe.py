import mii

model = '/data/xiaoxiawu/zhen/DeepSpeedExamples/benchmarks/inference/mii/fp16/mistralai/Mixtral-8x7B-v0.1'
# pipe = mii.pipeline(model, quantization_mode='wf6af
pipe = mii.pipeline(model, tensor_parallel=1, quantization_mode='wf6af16')

response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)


# pipe = mii.pipeline("NousResearch/Llama-2-70b-hf", quantization_mode='wf6af16')
# response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
# print(response)