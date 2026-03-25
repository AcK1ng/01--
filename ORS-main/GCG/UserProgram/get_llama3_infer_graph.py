#!/bin/env python


from typing import List, Optional
from llama3_8b_instruct_infer import Dialog, Llama
import os
import torch

ckpt_dir: str = "llama3_8b_instruct_infer/"
tokenizer_path: str = "llama3_8b_instruct_infer/tokenizer.model"
temperature: float = 0.6
top_p: float = 0.9
max_seq_len: int = 16*1024

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    temperature=temperature,
    top_p=top_p
)

def sync_all_gpus():
    if torch.cuda.device_count() > 0:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=torch.device(f'cuda:{device_id}'))

import time
kvcaches = generator.generate_kvcaches(128)
# 预热
for _ in range(500):
   token = generator.my_generate(kvcaches, 0, [0] * 1, 1, use_nvtx=False)
torch.cuda.synchronize()

# for _ in range(1):
#     s = time.time()
#     for _ in range(1000):
#         token = generator.my_generate(kvcaches, 0, [0] * 1, 1, use_nvtx=True)
#         sync_all_gpus()
#     e = time.time()
#     print("Time: ", e-s)
# exit()

for _ in range(1):
    s = time.time()
    for _ in range(1000):
        token = generator.my_generate(kvcaches, 0, [0] * 1, 1, use_nvtx=False)
        sync_all_gpus()
    e = time.time()
    print("Time: ", e-s)
exit()


import my_graph_capture

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

shape_env = ShapeEnv()

with FakeTensorMode(static_shapes = False, shape_env = shape_env) as fake_mode:
    ep = generator.get_computational_graph(shape_env)

    parameter_partial = [
        
        ["p_model_layers_0_",
        "p_model_layers_1_",
        "p_model_layers_2_",
        "p_model_layers_3_",
        "p_model_layers_4_",
        "p_model_layers_5_",
        "p_model_layers_6_",
        "p_model_layers_7_",],
        ["p_model_layers_8_",
        "p_model_layers_9_",
        "p_model_layers_10_",
        "p_model_layers_11_",
        "p_model_layers_12_",
        "p_model_layers_13_",
        "p_model_layers_14_",
        "p_model_layers_15_",],
        ["p_model_layers_16_",
        "p_model_layers_17_",
        "p_model_layers_18_",
        "p_model_layers_19_",
        "p_model_layers_20_",
        "p_model_layers_21_",
        "p_model_layers_22_",
        "p_model_layers_23_",],
        ["p_model_layers_24_",
        "p_model_layers_25_",
        "p_model_layers_26_",
        "p_model_layers_27_",
        "p_model_layers_28_",
        "p_model_layers_29_",
        "p_model_layers_30_",
        "p_model_layers_31_",]


    ]

    intermediate = my_graph_capture.capture_from_torch_export__stage1(ep, 
                                                                    #   parameter_partial
                                                                      )

graph_pack = my_graph_capture.capture_from_torch_export__stage2(intermediate)

local_hit = 3
miss_len = 12
kvcaches = generator.generate_kvcaches(128)
tokens = torch.zeros((1, miss_len,), dtype=torch.long)
tensor__for_local_hit = torch.empty((local_hit,))
example_inputs = (kvcaches + (tokens, tensor__for_local_hit,))
graph_pack["example_inputs"] = example_inputs


my_graph_capture.save_without_states_to(graph_pack, "llama3_infer_TS")
