#!/bin/env python

import my_graph_capture
import time
import torch

llama3_infer = my_graph_capture.GCGProgram("127.0.0.1:6067")

graph_pack = my_graph_capture.read_from_no_states("llama3_infer_TS")

fw_first_params_ids = llama3_infer.new_empty_tensors(graph_pack["fw_params_shapes"])
fw_buffer_ids = [llama3_infer.upload_tensor(t) for t in graph_pack["fw_buffer"]]

symbol__to__symexpr = graph_pack["symbol__to__symexpr"]
infer_task_id = llama3_infer.submit_graph(graph_pack["ts_fw_root"], graph_pack["ts_fw_submods"], symbol__to__symexpr)

infer_assignment = [0,0,0,0,0,0,0,0,0,0,0,]

def gen_kvcache(max_seq_len, placement: int):
    shape = torch.Size([2, max_seq_len, 8, 128])
    kvcache_tensor_descriptor = [(shape, torch.float, torch.strided) for _ in range(32)]
    return llama3_infer.new_empty_tensors(kvcache_tensor_descriptor, placement)


ranks = [0]


first_sleep = 0
context_len = 1
gen_token = 1
rank = 0
tokens_descriptor = [(torch.Size([1, context_len]), torch.long, torch.strided)]
# tokens_id = llama3_infer.new_empty_tensors(tokens_descriptor)
tokens_id = llama3_infer.new_empty_tensors(tokens_descriptor, rank)
local_hit_descriptor = [(torch.Size([0]), torch.float, torch.strided)]
# local_hit_descriptor_id = llama3_infer.new_empty_tensors(local_hit_descriptor)
local_hit_descriptor_id = llama3_infer.new_empty_tensors(local_hit_descriptor, rank)
kvcache_ids = gen_kvcache(128, rank)
# infer_assignment__ = infer_assignment
infer_assignment__ = [rank] * len(infer_assignment)
for _ in range(1001):
    ret = llama3_infer.run_task(infer_task_id, fw_first_params_ids + fw_buffer_ids + kvcache_ids + tokens_id + local_hit_descriptor_id, infer_assignment__, [0])
    llama3_infer.drop_future(ret[0])
    llama3_infer.drop_future(ret[1])
pass