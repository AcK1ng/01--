import os
import torch
import numpy as np

class EmptyClass:
    pass
configs = EmptyClass()
configs.task_name = 'long_term_forecast'
configs.features = 'MS'
configs.seq_len = 168
configs.pred_len = 24
configs.use_norm = 1
configs.patch_len = 24
configs.enc_in = 3
configs.d_model = 512
configs.dropout = 0.1
configs.embed = 'timeF'
configs.freq = 'h'
configs.e_layers = 3
configs.factor = 1
configs.n_heads = 8
configs.d_ff = 512
configs.activation = 'gelu'
from TimeXer import TimeXer
model = TimeXer.Model(configs).to("cuda")
example_input = torch.load("TimeXer/example_input.pth")
def states_to_device_as_example_inputs(states):
    device = "cuda"
    flatted, tree_spec = torch.utils._pytree.tree_flatten(states)
    device_placed_flatted = [t.to(device) if isinstance(t, torch.Tensor) else t for t in flatted]
    inputs_t = torch.utils._pytree.tree_unflatten(device_placed_flatted, tree_spec)
    return inputs_t
example_input = states_to_device_as_example_inputs(example_input)
output = model(*example_input)
pass

Gmodel = None
def my_compiler(gm: torch.fx.GraphModule, example_inputs):
    global Gmodel
    Gmodel = gm
    return gm.forward
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True
fmodel = torch.compile(model = model, backend = my_compiler, fullgraph=True, dynamic=True)
l=fmodel(*example_input)
pass