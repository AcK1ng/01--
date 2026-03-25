#!/bin/env python

import GCGUserprogram
import time
import torch

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(script_dir, '../GCGScheduler/build/Test')
sys.path.append(package_path)
import PythonGCG

master = PythonGCG.GetSimpleFullMaster()
cluster = PythonGCG.GetNativeCANNCPUCluster()

llama3_infer = GCGUserprogram.GCGProgramPythonPort(PythonGCG.ClusterWrapper(cluster, master))

graph_pack = GCGUserprogram.read_from_no_states("llama3_infer_TS")

fw_first_params_ids = llama3_infer.new_empty_tensors(graph_pack["fw_params_shapes"])
fw_buffer_ids = [llama3_infer.upload_tensor(t) for t in graph_pack["fw_buffer"]]

symbol__to__symexpr = graph_pack["symbol__to__symexpr"]
infer_task_id = llama3_infer.submit_graph(graph_pack["ts_fw_root"], graph_pack["ts_fw_submods"], symbol__to__symexpr)

infer_assignment = [0,0,0,0,0,0,0,0,0,0,0,]

def gen_kvcache(max_seq_len, placement: int):
    shape = torch.Size([2, max_seq_len, 8, 128])
    kvcache_tensor_descriptor = [(shape, torch.float, torch.strided) for _ in range(32)]
    return llama3_infer.new_empty_tensors(kvcache_tensor_descriptor, placement)


import time

warm_ranks = [0,1,2,3,4,5,6,7]
for rank in warm_ranks:
    tokens_descriptor = [(torch.Size([1, 1024]), torch.long, torch.strided)]
    tokens_id = llama3_infer.new_empty_tensors(tokens_descriptor, rank)
    local_hit_descriptor = [(torch.Size([0]), torch.float, torch.strided)]
    local_hit_descriptor_id = llama3_infer.new_empty_tensors(local_hit_descriptor, rank)
    kvcache_ids = gen_kvcache(8*1024, rank)

    infer_assignment__ = [rank] * len(infer_assignment)
    ret = llama3_infer.run_task(infer_task_id, fw_first_params_ids + fw_buffer_ids + kvcache_ids + tokens_id + local_hit_descriptor_id, infer_assignment__, [0])
    
    llama3_infer.drop_future(ret[0])
    llama3_infer.drop_future(ret[1])
    llama3_infer.drop_future(tokens_id[0])
    llama3_infer.drop_future(local_hit_descriptor_id[0])
    for id in kvcache_ids:
        llama3_infer.drop_future(id)

import time
time.sleep(60)



def new_req(context_len, gen_token, max_seq_len, rank):

    tokens_descriptor = [(torch.Size([1, context_len]), torch.long, torch.strided)]
    # tokens_id = llama3_infer.new_empty_tensors(tokens_descriptor)
    tokens_id = llama3_infer.new_empty_tensors(tokens_descriptor, rank)

    local_hit_descriptor = [(torch.Size([0]), torch.float, torch.strided)]
    # local_hit_descriptor_id = llama3_infer.new_empty_tensors(local_hit_descriptor)
    local_hit_descriptor_id = llama3_infer.new_empty_tensors(local_hit_descriptor, rank)

    kvcache_ids = gen_kvcache(8*1024, rank)
    # infer_assignment__ = infer_assignment
    infer_assignment__ = [rank] * len(infer_assignment)

    for _ in range(gen_token):
        ret = llama3_infer.run_task(infer_task_id, fw_first_params_ids + fw_buffer_ids + kvcache_ids + tokens_id + local_hit_descriptor_id, infer_assignment__, [0])

        llama3_infer.drop_future(tokens_id[0])
        llama3_infer.drop_future(local_hit_descriptor_id[0])
        tokens_id = [ret[0]]
        local_hit_descriptor_id = [ret[1]]

    for id in kvcache_ids:
        llama3_infer.drop_future(id)


import time
def callback__for_req_issuing(namely_time, contextlen, generated):
    rank = 0
    new_req(int(contextlen), int(generated), 8192, rank)
    print(f"At time {time.time()}, {namely_time} issue request with contextlen {contextlen} and generate {generated} Rank{rank}", flush=True)

def callback__for_req_issuing1(namely_time, contextlen, generated):
    rank = 1
    new_req(int(contextlen), int(generated), 8192, rank)
    print(f"At time {time.time()}, {namely_time} issue request with contextlen {contextlen} and generate {generated} Rank{rank}", flush=True)



dataset = GCGUserprogram.AzureDataTrace("AzurePublicDataset/data/Pressure.csv",
                                          callback__for_req_issuing,
                                          1)

dataset1 = GCGUserprogram.AzureDataTrace("AzurePublicDataset/data/Pressure.csv",
                                          callback__for_req_issuing1,
                                          5)
dataset.run()
dataset1.run()



while True:
    time.sleep(1)
