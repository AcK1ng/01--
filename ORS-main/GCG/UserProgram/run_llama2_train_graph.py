#!/bin/env python

import my_graph_capture
import time

llama2_infer = my_graph_capture.GCGProgram("127.0.0.1:6063")

graph_pack = my_graph_capture.read_from_no_states("llama2_grads_TS")

symbol__to__symexpr = graph_pack["symbol__to__symexpr"]

infer_assignment = [0,0,0,0,0,0,0,0,0,1,1,1,1,]
infer_task_id = llama2_infer.submit_graph(graph_pack["ts_fw_root"], graph_pack["ts_fw_submods"], symbol__to__symexpr)


train_assignment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,]
train_task_id = llama2_infer.submit_graph(graph_pack["ts_grad_root"], graph_pack["ts_grad_submods"], symbol__to__symexpr)

fw_first_params_ids = llama2_infer.new_empty_tensors(graph_pack["fw_params_shapes"])

fw_buffer_ids = [llama2_infer.upload_tensor(t) for t in graph_pack["fw_buffer"]]

example_inputs_ids = [llama2_infer.upload_tensor(t) for t in graph_pack["example_inputs"]]

inputs = fw_first_params_ids + fw_buffer_ids + example_inputs_ids

experment = 1
if experment == 0:
    for _ in range(1):
        output_futures = llama2_infer.run_task(infer_task_id, fw_first_params_ids + fw_buffer_ids + example_inputs_ids, infer_assignment)
        print("Infer Output Futures: ", output_futures)
        #for future in output_futures:
        #    llama2_infer.drop_future(future)
    pass
elif experment == 1:
    for _ in range(32):
        output_futures = llama2_infer.run_task(train_task_id, fw_first_params_ids + fw_buffer_ids + example_inputs_ids, train_assignment)
        trained_params, out = output_futures[:len(fw_first_params_ids)], output_futures[len(fw_first_params_ids):]
        for future in out + fw_first_params_ids:
           llama2_infer.drop_future(future)
        fw_first_params_ids = trained_params

    j = llama2_infer.get_master_status()

    j["train_task_id"] = train_task_id
    j["fw_first_params_ids"] = fw_first_params_ids
    j["fw_buffer_ids"] = fw_buffer_ids
    j["example_inputs_ids"] = example_inputs_ids
    j["train_assignment"] = train_assignment

    import json
    with open('master.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(j, indent = 2))
        
    pass
elif experment == 2:
    pass
else:
    exit()


while True:
    time.sleep(1)
