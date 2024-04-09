import torch
import tensorrt_llm
hidden_size = 512
output_size = 64
symmetric_quantizer = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix
weight = torch.randn(hidden_size, output_size, dtype=torch.float16, device='cuda')
act_torch_weights_fc1, torch_weight_scales_fc1 = symmetric_quantizer(weight.cpu().contiguous(), torch.quint4x2)

print("!!!size", weight.size(), act_torch_weights_fc1.size(), torch_weight_scales_fc1.size())


lower = (act_torch_weights_fc1 << 4) >> 4  # Arithmetic right shift sign extends
upper = (act_torch_weights_fc1 >> 4)
quant_weights =  torch.stack((lower, upper), dim=1).view(weight.shape) #torch.stack((lower, upper), dim=2)#
result = torch.multiply(quant_weights,
                            torch_weight_scales_fc1).contiguous().cuda()
print ("quantization error int4", torch.norm(weight-result))
print (weight.size(), quant_weights.size(),quant_weights.dtype)


act_torch_weights_fc1, torch_weight_scales_fc1 = symmetric_quantizer(weight.cpu().contiguous(), torch.int8)

print("!!!size", weight.size(), act_torch_weights_fc1.size(), torch_weight_scales_fc1.size())
result = (act_torch_weights_fc1*torch_weight_scales_fc1).contiguous().cuda()
print ("quantization error int8", torch.norm(weight-result))
print (weight.size(), quant_weights.size(),quant_weights.dtype)