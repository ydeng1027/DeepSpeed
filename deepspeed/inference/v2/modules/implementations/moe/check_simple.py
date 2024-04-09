import torch
import tensorrt_llm
hidden_size = 64
output_size = 512

weight = torch.randn(hidden_size, output_size, dtype=torch.float32, device='cuda')

# Quantize the floating point tensor to torch.quint4x2
# quantized_tensor = torch.quantize_per_tensor(weight, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda(), torch.quint4x2)

# # Dequantize back to floating point
# dequantized_tensor = quantized_tensor.dequantize()

# symmetric_quantizer = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix
# torch_transpose = torch.transpose(weight, 1, 2).contiguous().cpu()

# act_torch_weights_fc1, torch_weight_scales_fc1 = symmetric_quantizer(weight.cpu().contiguous(), torch.quint4x2)

# print("!!!size", weight.size(), act_torch_weights_fc1.size(), torch_weight_scales_fc1.size())


# lower = (act_torch_weights_fc1 << 4) >> 4  # Arithmetic right shift sign extends
# upper = (act_torch_weights_fc1 >> 4)
# quant_weights =  torch.stack((lower, upper), dim=1).view(weight.shape) #torch.stack((lower, upper), dim=2)#
# result = torch.multiply(quant_weights,
#                             torch_weight_scales_fc1).contiguous().cuda()
# print ("quantization error int4", torch.norm(weight-result))
# print (weight.size(), quant_weights.size(),quant_weights.dtype)

# weight = torch.randn(hidden_size, output_size, dtype=torch.float32, device='cuda')




quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.T.cpu().contiguous(), torch.int8)

print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size())
quant_weights = quant_weights.to(dtype=weight.dtype)
result = torch.multiply(quant_weights,
                            torch_weight_scales_fc1.unsqueeze(0)).T.contiguous()
print (result)
print ("quantization error int8", torch.norm(weight-result.to(device=weight.device)))
print (weight.size(), quant_weights.size(),quant_weights.dtype)


print ("---------------------------------------")
quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.T.cpu().contiguous(), torch.quint4x2)
print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size())
    # Unpack the int4s int int8s
upper = (quant_weights >> 4)
lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
    
quant_weights = torch.stack((lower, upper), dim=2).view(weight.T.shape)
result = torch.multiply(quant_weights,
                            torch_weight_scales_fc1.unsqueeze(0)).T.contiguous()
print (result)
print ("quantization error int4", torch.norm(weight-result.to(device=weight.device)))
print (weight.size(), quant_weights.size(),quant_weights.dtype)




print ("---------------------------------------")

quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.cpu().contiguous(), torch.int8)

print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size(), torch_weight_scales_fc1.unsqueeze(0).size())
quant_weights = quant_weights.to(dtype=weight.dtype)
result = torch.multiply(quant_weights,
                            torch_weight_scales_fc1.unsqueeze(0)).contiguous()
print (result)
print ("quantization error int8", torch.norm(weight-result.to(device=weight.device)))
print (weight.size(), quant_weights.size(),quant_weights.dtype)

print ("---------------------------------------")

quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.cpu().contiguous(), torch.quint4x2)
upper = (quant_weights >> 4)
lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
    
quant_weights = torch.stack((lower, upper), dim=2).view(weight.shape)
print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size(), torch_weight_scales_fc1.unsqueeze(0).size())
quant_weights = quant_weights.to(dtype=weight.dtype)
result = torch.multiply(quant_weights,
                            torch_weight_scales_fc1.unsqueeze(0)).contiguous()
print (result)
print ("quantization error int8", torch.norm(weight-result.to(device=weight.device)))
print (weight.size(), quant_weights.size(),quant_weights.dtype)



# print ("---------------------------------------")
# num_expert = 8
# weight = torch.randn(num_expert, hidden_size, output_size, dtype=torch.float32, device='cuda')

# quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.cpu().contiguous(), torch.int8)

# print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size(), torch_weight_scales_fc1.unsqueeze(1).size())
# quant_weights = quant_weights.to(dtype=weight.dtype)
# result = torch.multiply(quant_weights,
#                             torch_weight_scales_fc1.unsqueeze(1)).contiguous()
# print (result)
# print ("quantization error int8", torch.norm(weight-result.to(device=weight.device)))
# print (weight.size(), quant_weights.size(),quant_weights.dtype)


# quant_weights,_, torch_weight_scales_fc1 =  torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weight.cpu().contiguous(), torch.int8)

# print("!!!size", weight.size(), quant_weights.size(), torch_weight_scales_fc1.size(), torch_weight_scales_fc1.unsqueeze(1).size())
# quant_weights = quant_weights.to(dtype=weight.dtype)
# result = torch.multiply(quant_weights,
#                             torch_weight_scales_fc1.unsqueeze(1)).contiguous()
# print (result)
# print ("quantization error int8", torch.norm(weight-result.to(device=weight.device)))
# print (weight.size(), quant_weights.size(),quant_weights.dtype)