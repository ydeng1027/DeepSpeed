import torch
from torch.utils.cpp_extension import load

# Load the C++ extension
moe_gemm_ext = load(name='moe_gemm_ext', sources=['moe_gemm.cu'], verbose=True)

# Define the input tensors
batch_size = 2
num_experts = 4
hidden_size = 128
output_size = 256

hidden_states = torch.randn(batch_size, hidden_size, dtype=torch.float16, device='cuda')
weight = torch.randn(num_experts, hidden_size, output_size, dtype=torch.int8, device='cuda')
scales = torch.randn(num_experts, 1, dtype=torch.float16, device='cuda')
bias = torch.randn(num_experts, output_size, dtype=torch.float16, device='cuda')
total_rows_before_expert = torch.tensor([0, batch_size // 2, batch_size], dtype=torch.int64, device='cuda')

# Create the output tensor
output = torch.empty(batch_size, output_size, dtype=torch.float16, device='cuda')

# Call the mixed_moe_gemm kernel
moe_gemm_ext.mixed_moe_gemm(output, hidden_states, weight, scales, bias, total_rows_before_expert, 8, 0)