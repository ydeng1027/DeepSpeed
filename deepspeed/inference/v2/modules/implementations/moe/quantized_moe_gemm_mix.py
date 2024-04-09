# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional, Tuple

import torch

from deepspeed.accelerator import get_accelerator
from ....allocator import empty_from
from ....inference_utils import ActivationType, is_gated
from ....kernels.core_ops import BlasLibLinear, CUDAGatedActivation
from ....kernels.ragged_ops import (
    MoEGather,
    MoEScatter,
    RaggedTopKGating,
)
from ....ragged import RaggedBatchWrapper

from ...interfaces import DSMoEBase, DSMoERegistry
from ...configs import DSMoEConfig
from ....kernels.cutlass_ops import MoEGEMM, MixedMoEGEMM
from ....inference_parameter import InferenceParameter
import tensorrt_llm 

# def int_quantize(input: torch.FloatTensor,
#                 num_bits: int = 8,
#                 min_value: torch.FloatTensor = None,
#                 max_value: torch.FloatTensor = None,
#                 group_size: int = -1):
#     """
#     Args:
#         inputs (`torch.FloatTensor`)
#             The input which needs to be quantized
#         num_bits (int, >=4)
#             Number of bits to use for quantization
#         min_value/max_vlue (torch.FloatTensor)
#             Used for static activation quantization
#         group_size (int) N
#             The quantization block size, each N numbers has its own scaling
#             factor and off-site. -1 means use the last dim as the group_size
#     Returns:
#         quantized_fake_fp6
#             The quantized weights, in fp16 format and contains fp6 value.
#         scales
#             Quantization scales
#     """

#     q_range = 2**num_bits
#     assert (min_value is None and max_value is None) or (min_value is not None and max_value is not None)

#     assert input.dtype == torch.float16

#     orig_device = input.device
#     input = input.to(torch.float32).to(get_accelerator().current_device())

#     input_shape = input.shape

#     if group_size == -1:
#         group_size = input_shape[-1]
#     else:
#         # Only support per-channel quantization
#         raise NotImplementedError
#     num_groups = input.numel() // group_size
#     input = input.reshape(num_groups, -1)

#     if min_value is None:
#         #min_value = input.amin(dim=-1, keepdim=True)
#         max_value = input.amax(dim=-1, keepdim=True)
        
#     scales = 2 * (max_value) / q_range
#     scales[scales == 0] = 1 
#     # zero_point = (min_value / scale).round() * scale   
#     #print ("check size",scales.size())
#    # print (torch.abs(((input / scales).round().clamp(-q_range // 2, q_range // 2 - 1)).reshape(input_shape)*scales-input).mean())
#     output = ((input / scales).round().clamp(-q_range // 2, q_range // 2 - 1)).reshape(input_shape).contiguous() 
#     #.to(torch.float16).to(orig_device)
#     return output, scales

index = 0
@DSMoERegistry.register_module
class DSINTmixedMultiGemmMoE(DSMoEBase):
    """
    MoE implementation based on the CUTLASS multi-GEMM.
    """

    @staticmethod
    def name():
        return 'int4_8_multi_gemm_moe'

    @staticmethod
    def supports_config(config: DSMoEConfig) -> bool:
        if config.input_dtype != config.output_dtype:
            return False

        if config.input_dtype != torch.float16 and config.input_dtype != torch.bfloat16:
            return False

        if config.top_k != 1 and config.top_k != 2:
            return False

        return True

    def __init__(self, config: DSMoEConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

        # Convenience variables for frequently accessed items.
        self.max_tokens = self._config.max_tokens
        self.n_experts = self._config.n_experts
        self.n_top_k = self._config.top_k
        self.intermediate_dim = self._config.intermediate_features

        moe_op_act_fn = ActivationType.IDENTITY if is_gated(self._config.activation) else self._config.activation
        
        #if index %2==0:
        self._mlp_1 = MixedMoEGEMM(fp_dtype=implementation_config['weight_dtype'], act_fn=moe_op_act_fn, num_bits=4)
        self._mlp_2 = MixedMoEGEMM(fp_dtype=implementation_config['weight_dtype'], act_fn=ActivationType.IDENTITY, num_bits=8)
        # self._mlp_1a = MoEGEMM(fp_dtype=implementation_config['weight_dtype'], act_fn=moe_op_act_fn, num_bits=8)
        #, num_bits=8
        if is_gated(self._config.activation):
            self._activation = CUDAGatedActivation(self._config.model_dim, self._config.input_dtype,
                                                   self._config.activation)
        else:
            self._activation = None

        self._gate_proj = BlasLibLinear(self._config.input_dtype)
        self._top_1_gate = RaggedTopKGating(config.input_dtype)
        self._moe_scatter = MoEScatter(config.input_dtype, config.model_dim)
        self._moe_gather = MoEGather(config.input_dtype, config.model_dim, config.normalize_scores)

        self._create_buffers()

    def _create_buffers(self):

        # Gating buffers
        self._logits = torch.empty((self._config.max_tokens, self.n_experts),
                                   dtype=self._config.input_dtype,
                                   device=get_accelerator().current_device())
        self._expert_counts = torch.empty((self.n_experts, ),
                                          dtype=torch.int32,
                                          device=get_accelerator().current_device())
        self._scores = torch.empty((self._config.max_tokens, self.n_top_k),
                                   dtype=torch.float32,
                                   device=get_accelerator().current_device())
        self._assignments = torch.empty((self._config.max_tokens, self.n_top_k),
                                        dtype=torch.int32,
                                        device=get_accelerator().current_device())
        self._offsets = torch.empty((self._config.max_tokens, self.n_top_k),
                                    dtype=torch.int32,
                                    device=get_accelerator().current_device())

        # Scatter buffers
        self._moe_input = torch.empty((self._config.max_tokens * self.n_top_k, self._config.model_dim),
                                      dtype=self._config.input_dtype,
                                      device=get_accelerator().current_device())
        self._expert_cumsum = torch.empty((self._config.n_experts, ),
                                          dtype=torch.int64,
                                          device=get_accelerator().current_device())
        self._mapped_slots = torch.empty((self._config.max_tokens, self.n_top_k),
                                         dtype=torch.int32,
                                         device=get_accelerator().current_device())

        # GEMM Buffers
        self._intermediate = torch.empty((self._config.max_tokens * self.n_top_k, self._config.intermediate_features),
                                         dtype=self._config.output_dtype,
                                         device=get_accelerator().current_device())
        if self._activation is not None:
            self._gated_intermediate = torch.empty(
                (self._config.max_tokens * self.n_top_k, self._config.intermediate_features * 2),
                dtype=self._config.output_dtype,
                device=get_accelerator().current_device())

        self._output_unordered = torch.empty((self._config.max_tokens * self.n_top_k, self._config.model_dim),
                                             dtype=self._config.output_dtype,
                                             device=get_accelerator().current_device())

        # Gather buffer
        self._output = torch.empty((self._config.max_tokens, self._config.model_dim),
                                   dtype=self._config.output_dtype,
                                   device=get_accelerator().current_device())

    def transform_gate_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Ensures gate param is going to match the activation data type.
        """
        param = param.to(self._config.input_dtype)
        return InferenceParameter.initialize(param)

    def transform_moe_mlp_2_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """

        param = param.to(self._config.input_dtype)
        if len(param.shape) == 2:
            # skip for bias tensor
            return

        if len(param.shape) == 3:
            param = param.permute(0, 2, 1).contiguous()
         # weight is [b, m, n], scales is [b, n]
        #param = torch.rand_like(param,dtype=torch.quint4x2)
        # weight_4bit = torch.quantize_per_tensor(param.to(torch.float32), scale=1.0, zero_point=0, dtype=torch.quint4x2)
        # scales = torch.rand((param.shape[0], param.shape[2]), dtype=torch.float16)
        # print ("check, here! moe_mlp_1", param.size())         
        symmetric_quantizer = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix
        #weight_8bit, scales = symmetric_quantizer(param.cpu(), torch.int8)
        weight_8bit, scales = symmetric_quantizer(param.cpu().contiguous(), torch.int8)
        print ("MLP2", weight_8bit.size(),scales.size()) 
        #MLP2 torch.Size([8, 14336, 4096]) torch.Size([8, 4096])
        return  InferenceParameter.initialize(weight_8bit, scales=scales)

    def transform_moe_mlp_1_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        global index
        index += 1
        param = param.to(self._config.input_dtype)
        if len(param.shape) == 2:
            # skip for bias tensor
            return

        if len(param.shape) == 3:
            param = param.permute(0, 2, 1).contiguous()
        # weight is [b, m, n], scales is [b, n]
        # Create a random tensor with the same shape as 'param' and data type torch.float32
        print ("INT4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~layer",index)
        # symmetric_quantizer = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix
        # weight_8bit, scales = symmetric_quantizer(param.cpu(), torch.int8)
        symmetric_quantizer = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix
        #weight_8bit, scales = symmetric_quantizer(param.cpu(), torch.int8)
        weight_4bit, scales = symmetric_quantizer(param.cpu().contiguous(), torch.quint4x2)   
        print ("MLP1", weight_4bit.size(),scales.size())    
        ##MLP1 torch.Size([8, 4096, 14336]) torch.Size([8, 28672]) 
        return InferenceParameter.initialize(weight_4bit, scales=scales)


    @property
    def output(self) -> torch.Tensor:
        return self._output

    def _gate(self, hidden_states: torch.Tensor, batch_metadata: RaggedBatchWrapper,
              gate_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper function to isolate the logit for gating. This will take the hidden states
        and produce the metadata + tensors for the CUTLASS ragged GEMMs. If the input has
        been padded for CG, this will strip the padding for MoE.

        Parameters:
            hidden_states (torch.Tensor): Hidden states tensor. Expected shape is [n_tokens, model_dim].
            batch_metadata (RaggedBatchWrapper): Batch metadata for the hidden states.
            gate_w (torch.Tensor): Gate weight tensor. Expected shape is [num_experts, model_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The MoE input, the cumsum of the offsets (for the MoE kernels themselves), the scores, and the mapped slots (to recover the original order of the tokens)
        """

        # Get views on the buffers for gating
        logits = empty_from(self._logits, (hidden_states.shape[0], self._logits.shape[-1]))
        scores = empty_from(self._scores, (hidden_states.shape[0], self.n_top_k))
        assignments = empty_from(self._assignments, (hidden_states.shape[0], self.n_top_k))
        offsets = empty_from(self._offsets, (hidden_states.shape[0], self.n_top_k))
        mapped_slots = empty_from(self._mapped_slots, (hidden_states.shape[0], self.n_top_k))
        moe_input = empty_from(self._moe_input, (hidden_states.shape[0] * self.n_top_k, self._moe_input.shape[-1]))

        self._gate_proj(logits, hidden_states, gate_w)
        self._expert_counts.zero_()
        self._top_1_gate(self._expert_counts, scores, assignments, offsets, logits, batch_metadata)
        self._moe_scatter(moe_input, self._expert_cumsum, mapped_slots, hidden_states, self._expert_counts,
                          assignments, offsets)

        return moe_input, self._expert_cumsum, scores, mapped_slots

    def forward(self,
                hidden_states: torch.Tensor,
                batch_metadata: RaggedBatchWrapper,
                gate_w: torch.Tensor,
                mlp_1_w: torch.Tensor,
                mlp_2_w: torch.Tensor,
                mlp_1_b: Optional[torch.Tensor] = None,
                mlp_2_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        MoE forward pass built on top of CUTLASS multi-GEMM.

        Parameters:
            hidden_states (torch.Tensor): Hidden states tensor. Expected shape is [batch, seq_len, model_dim].
            gate_w (torch.Tensor): Gate weight tensor. Expected shape is [num_experts, model_dim].
        """
        # global index
        # index += 1
        # self.index += 1
        moe_input, expert_cumsum, scores, mapped_slots = self._gate(hidden_states, batch_metadata, gate_w)

        # Get views on the buffers for GEMM
        intermediate = empty_from(self._intermediate,
                                  (hidden_states.shape[0] * self.n_top_k, self._intermediate.shape[-1]))
        output_unordered = empty_from(self._output_unordered,
                                      (hidden_states.shape[0] * self.n_top_k, self._output_unordered.shape[-1]))
        output = empty_from(self._output, (hidden_states.shape[0], self._output.shape[-1]))
        #import pdb;pdb.set_trace()
        if self._activation is not None:
            gated_intermediate = empty_from(self._gated_intermediate, (hidden_states.shape[0] * self.n_top_k, self._gated_intermediate.shape[-1]))
            self._mlp_1(gated_intermediate, moe_input, mlp_1_w, mlp_1_w.scales, expert_cumsum, mlp_1_b,)
                ### [54, 3584]    [54, 4096]    [8, 4096, 3584]  #[8, 3584] #[ 6, 15, 24, 30, 37, 43, 48, 54] None
            self._activation(intermediate, gated_intermediate)
        else:
            self._mlp_1(
                intermediate, ###[54, 1792]
                moe_input,
                mlp_1_w,
                mlp_1_w.scales,
                expert_cumsum,
                mlp_1_b)
        self._mlp_2(output_unordered, intermediate, mlp_2_w, mlp_2_w.scales, expert_cumsum, mlp_2_b,)

        self._moe_gather(output, output_unordered, scores, mapped_slots, self._expert_counts)
        return output
