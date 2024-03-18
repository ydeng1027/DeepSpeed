import torch
from torch.utils.cpp_extension import load
import deepspeed
from deepspeed.inference.v2.kernels.cutlass_ops import MoEGEMM, MixedMoEGEMM
# Load the C++ extension
from deepspeed.inference.v2.inference_utils import ActivationType
mlp_1 = MixedMoEGEMM(fp_dtype=torch.float16, act_fn=ActivationType.IDENTITY, num_bits=8)
mlp_2 = MoEGEMM(fp_dtype=torch.float16, act_fn=ActivationType.IDENTITY)#, num_bits=8

def int_quantize(input: torch.FloatTensor,
                num_bits: int = 8,
                min_value: torch.FloatTensor = None,
                max_value: torch.FloatTensor = None,
                group_size: int = -1):
    """
    Args:
        inputs (`torch.FloatTensor`)
            The input which needs to be quantized
        num_bits (int, >=4)
            Number of bits to use for quantization
        min_value/max_vlue (torch.FloatTensor)
            Used for static activation quantization
        group_size (int) N
            The quantization block size, each N numbers has its own scaling
            factor and off-site. -1 means use the last dim as the group_size
    Returns:
        quantized_fake_fp6
            The quantized weights, in fp16 format and contains fp6 value.
        scales
            Quantization scales
    """

    q_range = 2**num_bits
    assert (min_value is None and max_value is None) or (min_value is not None and max_value is not None)

    assert input.dtype == torch.float16

    input = input.to(torch.float32).cuda()

    input_shape = input.shape

    if group_size == -1:
        group_size = input_shape[-1]
    else:
        # Only support per-channel quantization
        raise NotImplementedError
    num_groups = input.numel() // group_size
    input = input.reshape(num_groups, -1)

    if min_value is None:
        max_value = input.amax(dim=-1, keepdim=True)
        
    scales = 2 * (max_value) / q_range
    scales[scales == 0] = 1
    output = ((input / scales).round().clamp(-q_range // 2, q_range // 2 - 1)).reshape(input_shape).contiguous() 
    #.to(torch.float16).to(orig_device)
    print (scales.size())
    print (scales)
    return output, scales


# Define the input tensors
sqeuence = sum([0,1,2,3,4,5,6,7]) ##
num_experts = 8 ####
hidden_size = 512 ####
output_size = 768

hidden_states = torch.randn((sqeuence, hidden_size), dtype=torch.float16, device='cuda')

weight = torch.randn(num_experts, hidden_size, output_size, dtype=torch.float16, device='cuda')

weight_int8 = torch.zeros(num_experts, hidden_size, output_size, dtype=torch.int8, device='cuda')
scales = torch.ones(num_experts, output_size, dtype=torch.float16, device='cuda')
for i in range(num_experts):
    int8_w, int8_s = int_quantize(weight[i].T)
    weight_int8[i] = int8_w.to(torch.int8).T
    scales[i]  = int8_s.flatten()

import tensorrt_llm
symmetric_quantizer = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix
act_torch_weights_fc1, torch_weight_scales_fc1 = symmetric_quantizer(weight.cpu(), torch.int8)
print(torch_weight_scales_fc1)

#import pdb;pdb.set_trace()
print ("quantization error", torch.norm(weight-torch.multiply(weight_int8, scales.unsqueeze(1))))
print ("quantization error", torch.norm(weight-weight_int8*scales.unsqueeze(1)))
bias = torch.randn(num_experts, output_size, dtype=torch.float16, device='cuda')
total_rows_before_expert = torch.tensor([4,1,2,3,4,5,6,3], dtype=torch.int64, device='cuda')
total_rows_before_expert = total_rows_before_expert.cumsum(dim=0)

# print (total_rows_before_expert)
#print X=(xx, 512)==>[8, xx//8, 512], W=(8, 512, 768),  
print ("===================>", weight.size())
output3 = torch.empty(sqeuence, output_size, dtype=torch.float16, device='cuda')
mlp_2(output3, hidden_states, weight, total_rows_before_expert, None)


output2 = torch.empty(sqeuence, output_size, dtype=torch.float16, device='cuda')
mlp_2(output2, hidden_states, weight_int8*scales.unsqueeze(1), total_rows_before_expert, None)

output1 = torch.empty(sqeuence, output_size, dtype=torch.float16, device='cuda')
mlp_1(output1, hidden_states, act_torch_weights_fc1.cuda(), torch_weight_scales_fc1.cuda(), total_rows_before_expert,None)

index = 0
for x,y,z in zip(output1, output2, output3):
    index += 1
    print (index, torch.mean(x.abs()).item(), torch.mean(y.abs()).item(), torch.mean((x-y).abs()).item(),"-----",
           torch.mean(z.abs()).item(), torch.mean((y-z).abs()).item(), torch.mean((x-z).abs()).item())

