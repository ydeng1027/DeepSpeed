
import torch
import tensorrt_llm

num_expert = 8
hidden_size = 64*2
output_size = 512*2
weight = torch.randn(num_expert, hidden_size, output_size, dtype=torch.float32, device='cuda')
print (weight[0])
print ("---------------------------------------")
quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.cpu().contiguous(), torch.int8)

print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size(), torch_weight_scales_fc1.unsqueeze(1).size())
quant_weights = quant_weights.to(dtype=weight.dtype)
result = torch.multiply(quant_weights,
                            torch_weight_scales_fc1.unsqueeze(1)).contiguous()
print (result[0])
print ("quantization error int8", torch.norm(weight-result.to(device=weight.device)))
print (weight.size(), quant_weights.size(),quant_weights.dtype)


print ("===========================================")
quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.cpu().contiguous(), torch.quint4x2)
print ("!!!before size", quant_weights.size())
lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
upper = (quant_weights >> 4)
quant_weights = torch.stack((lower, upper), dim=3).view(weight.shape)
print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size(), torch_weight_scales_fc1.unsqueeze(1).size())
quant_weights = quant_weights.to(dtype=weight.dtype)
result = torch.multiply(quant_weights,
                            torch_weight_scales_fc1.unsqueeze(1)).contiguous()
print (result[0])
print ("quantization error int4", torch.norm(weight-result.to(device=weight.device)))
print (weight.size(), quant_weights.size(),quant_weights.dtype)



print ("===========================================")
weight = weight.reshape(8, 64, -1)
quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.cpu().contiguous(), torch.quint4x2)
print ("!!!before size", quant_weights.size())
lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
upper = (quant_weights >> 4)
quant_weights = torch.stack((lower, upper), dim=3).view(weight.shape)
print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size(), torch_weight_scales_fc1.unsqueeze(1).size())
quant_weights = quant_weights.to(dtype=weight.dtype)
result = torch.multiply(quant_weights,
                            torch_weight_scales_fc1.unsqueeze(1)).contiguous()
print (result.reshape(8, 64*2, -1)[0])
print ("quantization error int4", torch.norm(weight-result.to(device=weight.device)))
print (weight.size(), quant_weights.size(),quant_weights.dtype)