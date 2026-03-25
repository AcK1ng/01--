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

infer_assignment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,]
infer_assignment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6]
infer_assignment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6]
infer_assignment = [0,0,0,0,0,0,0,0,0,0,0,]

def gen_kvcache(max_seq_len, placement: int):
    shape = torch.Size([2, max_seq_len, 8, 128])
    kvcache_tensor_descriptor = [(shape, torch.float, torch.strided) for _ in range(32)]
    return llama3_infer.new_empty_tensors(kvcache_tensor_descriptor, placement)

ranks = [1,0,2,0,3,0,4,0,5,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
# ranks = [1,2,3,4,5,6]
ranks = [1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,]
# ranks = [0]
i = 0
def new_req(context_len, gen_token, max_seq_len):

    global i

    rank = ranks[i % len(ranks)]

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

    i += 1

def callback__for_req_issuing(t, contextlen, generated):
    new_req(int(contextlen), int(generated), 8192)
    print("At time {}, issue request with contextlen {} and generate {}".format(t, contextlen, generated))


dataset = my_graph_capture.AzureDataTrace("AzurePublicDataset/data/MyLLMTrace.csv",
                                          callback__for_req_issuing,
                                          0.1)

dataset.run()



while True:
    time.sleep(1)
