import torch
from torch.utils.cpp_extension import load
import deepspeed
from deepspeed.inference.v2.kernels.cutlass_ops import MoEGEMM, MixedMoEGEMM
# Load the C++ extension
from deepspeed.inference.v2.inference_utils import ActivationType
import numpy as np
mlp_0 = MixedMoEGEMM(fp_dtype=torch.float16, act_fn=ActivationType.IDENTITY, num_bits=8)
mlp_1 = MixedMoEGEMM(fp_dtype=torch.float16, act_fn=ActivationType.IDENTITY, num_bits=4)
mlp_2 = MoEGEMM(fp_dtype=torch.float16, act_fn=ActivationType.IDENTITY)#, num_bits=8

sqeuence = sum([0,1,2,3,4,5,6,7]) ##
num_experts = 8 ####
hidden_size = 2048 ####
output_size = 64

hidden_states = torch.randn((sqeuence, hidden_size), dtype=torch.float16, device='cuda')

weight = torch.randn(num_experts, hidden_size, output_size, dtype=torch.float16, device='cuda')
import tensorrt_llm
symmetric_quantizer = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix
weight_int8, scales = symmetric_quantizer(weight.cpu().contiguous(), torch.int8)
    
symmetric_quantizer = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix


# torch_transpose = torch.transpose(weight, 1, 2).contiguous().cpu()
act_torch_weights_fc1,  torch_weight_scales_fc1 = symmetric_quantizer(weight.cpu().contiguous(), torch.quint4x2)
#[8, 512, 128]     [8, 256]                                      #[8, 512, 256]
print("!!!size", weight.size(), act_torch_weights_fc1.size(), torch_weight_scales_fc1.size())
# act_torch_weights_fc1 =  torch.transpose(act_torch_weights_fc1, 2, 1).contiguous().cpu()
# torch_weight_scales_fc1 = torch_weight_scales_fc1.unsqeeze().contiguous().cpu()
# # print (act_torch_weights_fc1)
# lower = (act_torch_weights_fc1 << 4) >> 4  # Arithmetic right shift sign extends
# upper = (act_torch_weights_fc1 >> 4)

# quant_weights =  torch.stack((lower, upper), dim=3).view(weight.shape) #torch.stack((lower, upper), dim=2)#
# result = torch.multiply(quant_weights,  torch_weight_scales_fc1.unsqueeze(1)).contiguous().cuda()

# print ("quantization error 111111111111", torch.norm(weight-result))
# print (weight.size(), quant_weights.size(),quant_weights.type)

total_rows_before_expert = torch.tensor([4,1,2,3,4,5,6,3], dtype=torch.int64, device='cuda').cumsum(dim=0)

output3 = torch.empty(sqeuence, output_size, dtype=torch.float16, device='cuda')
mlp_2(output3, hidden_states, weight, total_rows_before_expert, None)

# output2 = output3
output2 = torch.empty(sqeuence, output_size, dtype=torch.float16, device='cuda')
mlp_0(output2, hidden_states, weight_int8.cuda(), scales.cuda(), total_rows_before_expert, None)
# 
# print ('quant_weights',quant_weights)
output1 = torch.empty(sqeuence, output_size, dtype=torch.float16, device='cuda')
print ("!!!!!!!!!!!!!!!!!!!!!!",act_torch_weights_fc1.size(2), output1.size(1))
mlp_1(output1, \
    hidden_states, \
    act_torch_weights_fc1.cuda(), \
    torch_weight_scales_fc1.cuda(), \
    total_rows_before_expert,None)

index = 0
for x,y,z in zip(output1, output2, output3):
    index += 1
    print (index, torch.mean(x.abs()).item(), torch.mean(y.abs()).item(), torch.mean((x-y).abs()).item(),"-----",
           torch.mean(z.abs()).item(), torch.mean((y-z).abs()).item(), torch.mean((x-z).abs()).item())

