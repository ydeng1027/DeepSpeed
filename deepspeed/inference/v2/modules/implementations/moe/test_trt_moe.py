
import tensorrt_llm
from tensorrt_llm import Tensor
import tensorrt as trt
from tensorrt_llm.layers.moe import MoeConfig
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm._utils import torch_to_numpy, trt_dtype_to_torch
import torch
import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

builder = tensorrt_llm.Builder()
hidden_size = 64
fc1_out_size = 64
ffn_hidden_size = 4 * hidden_size
num_experts = 4
quant_mode = QuantMode.use_weight_only(
                use_int4_weights=True)
actfn = 'gelu'
is_gated = False
weight_dtype = 'float16'
dtype = tensorrt_llm.str_dtype_to_trt(weight_dtype)
def gen_uniform_weights(*args, **kwargs):
    return (torch.rand(*args, **kwargs) * 2 - 1).contiguous()

genfn = gen_uniform_weights if weight_dtype == trt.int8 else torch.randn

fc1_out_size = ffn_hidden_size * 2 if is_gated else ffn_hidden_size

fc1_weights = genfn((num_experts, fc1_out_size, hidden_size),
                            dtype=trt_dtype_to_torch(dtype),
                        device="cuda")
fc2_weights = genfn((num_experts, hidden_size, ffn_hidden_size),
                            dtype=trt_dtype_to_torch(dtype),
                            device="cuda")
def set_weight_layer(input_weights, weight, scale, quant_mode):
    #print (input_weights.data.size(), weight.data.size())
    if quant_mode.is_weight_only():
        torch_transpose = torch.transpose(input_weights, 1,
                                            2).contiguous().cpu()
        print (quant_mode.is_int4_weight_only(),"input_weights", input_weights.size())
        type = torch.quint4x2 if quant_mode.is_int4_weight_only(
        ) else torch.int8
        print ('torch_transpose', torch_transpose.size(), type)
        processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
            torch_transpose, type)
        # Change the shape to what moe expects without touching the underlying format
        print ('plese check this.~!!!!! int4 or not')
        print (quant_mode.is_int4_weight_only(),weight.value.size(), processed_torch_weights.size())
        ### True, (4, 64, 128) (4, 64, 128)
        weight.value = np.ascontiguousarray(
            torch_to_numpy(processed_torch_weights))
        print ('~~~~~~~ weight scale', torch_weight_scales.size())
        ### True, (4, 256)
        scale.value = np.ascontiguousarray(
            torch_to_numpy(torch_weight_scales))
num_sequences = 1
sequence_lengths = 1024
input_data = gen_uniform_weights(
    (num_sequences, sequence_lengths, hidden_size),
    dtype=trt_dtype_to_torch(dtype))        
router_weights = torch.randn((num_experts, hidden_size),
                                    dtype=trt_dtype_to_torch(dtype),
                                    device="cuda")
builder = tensorrt_llm.Builder()
net = builder.create_network()
net.plugin_config.set_moe_plugin(dtype)
with tensorrt_llm.net_guard(net):
    network = tensorrt_llm.default_trtnet()
    moe = tensorrt_llm.layers.MOE(moe_config=MoeConfig(
                            num_experts=num_experts,
                            top_k=2,
                            normalization_mode=MoeConfig.ExpertScaleNormalizationMode.NONE),
                            hidden_size=hidden_size,
                            ffn_hidden_size=ffn_hidden_size,
                            hidden_act=actfn,
                            bias=False,
                            dtype=dtype,
                            quant_mode=quant_mode)
    moe.router.weight.value = torch_to_numpy(router_weights.cpu())
    set_weight_layer(fc1_weights, moe.experts_weight_1,
                                    moe.experts_scale_1, quant_mode)
    set_weight_layer(fc2_weights, moe.experts_weight_2,
                                    moe.experts_scale_2, quant_mode)

    trt_key = Tensor(name='input_hidden_states',
                        shape=tuple(input_data.shape),
                        dtype=dtype)
    finished = None
    trt_finished = Tensor(name='input_finished',
                            shape=tuple(finished.shape),
                            dtype=tensorrt_llm.str_dtype_to_trt(
                                'bool')) if finished is not None else None
    output = moe(trt_key, trt_finished).trt_tensor
    output.name = 'output'
    output.dtype = dtype
    network.mark_output(output)
    print (output)
    # import pdb;pdb.set_trace()

build_engine = EngineFromNetwork(
    (builder.trt_builder, net.trt_network),
    config=CreateConfig(fp16=True,#(dtype == trt.float16),
                        bf16=False, #(dtype == trt.bfloat16),
                        int8=True, #(weight_dtype == trt.int8),
                        precision_constraints='obey',
                        builder_optimization_level=4))
assert build_engine is not None

with TrtRunner(build_engine) as runner:
    feed_dict = {
        'input_hidden_states': input_data,
    }
    outputs = runner.infer(feed_dict=feed_dict)
    print (outputs['output'].size())