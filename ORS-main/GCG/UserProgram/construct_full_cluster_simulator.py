#!/bin/env python

import my_graph_capture
import time
import json

llama2_infer = my_graph_capture.GCGProgram("127.0.0.1:6061")

with open('master.json', 'r', encoding='utf-8') as file:
    master_status = file.read()

master_status_j = json.loads(master_status)

def get_thousand_gpu_rank_manager():
    model = 'NVIDIA A100 80GB PCIe'
    hbm_capability = 84974239744
    host_resource = 'cuda'
    nr_hosts = 1
    nr_rank__per_host = 8
    rank = 0
    
    bw_inner_host = 600 # MB per ms
    bw_between_hosts = 12.5 # MB per ms
    c = 1 # ms
    constant__for_small_block = 200 # ms
    
    rank_manager = {
        "graph": [],
        "host_models": [],
        "host_resources": [],
        "hosts": [],
        "nodes": [],
        "transmit_predictor": {'lr': 0.01,
                                'pair__to__model_id': [],
                                'ranks': [],
                                'type': 'LinearTransmitPerformancePredictor',
                                'warmup_step': 3}
    }
    
    rank__to__host_id = {}
    
    for host_id in range(nr_hosts):
        ranks_in_host = []
        for local_rank in range(nr_rank__per_host):
            node_info = {'host_id': host_id,
             'model': model,
             'rank': rank,
             'resource': host_resource,
             'sign_in_info':
                 {'acc_model':model,
                  'acc_name': f"{host_resource}:{local_rank}",
                  'compute_unit': 1,
                  'host_id': host_id,
                  'hbm_capability': hbm_capability,
                  'local_device_id': local_rank,
                  'never_signout': 1,
                  'rank': rank,
                  'resource': host_resource}}
            ranks_in_host.append(rank)
            rank__to__host_id[rank] = host_id
            rank_manager["graph"].append([rank, []])
            rank_manager["nodes"].append([rank, node_info])
            rank_manager["transmit_predictor"]['ranks'].append(rank)
            rank += 1
    
        rank_manager["host_models"].append([host_id, model])
        rank_manager["host_resources"].append([host_id, host_resource])
        rank_manager["hosts"].append([host_id, ranks_in_host])
    

    model_in_host = {'_bw': bw_inner_host,
             'constant': constant__for_small_block,
             'c': c,
             'lr': 0.01,
             'warmup_step': 0}
    model_between_hosts = {'_bw': bw_between_hosts,
             'constant': constant__for_small_block,
             'c': c,
             'lr': 0.01,
             'warmup_step': 0}
    rank_manager["transmit_predictor"]['models'] = [model_in_host, model_between_hosts]
    pair_models = set()
    for send in range(rank):
        for recv in range(rank):
            if send == recv:
                continue
            rank_pair = (min(send, recv), max(send, recv))
            r1, r2 = rank_pair
            if rank_pair in pair_models:
                continue
            if rank__to__host_id[send] == rank__to__host_id[recv]:
                rank_manager["transmit_predictor"]['pair__to__model_id'].append([[r1, r2], 0])
            else:
                rank_manager["transmit_predictor"]['pair__to__model_id'].append([[r1, r2], 1])
            pair_models.add(rank_pair)
    return rank_manager


# master_status_j["rank_manager"] = get_thousand_gpu_rank_manager()


train_task_id = master_status_j["train_task_id"]
fw_first_params_ids = master_status_j["fw_first_params_ids"]
fw_buffer_ids = master_status_j["fw_buffer_ids"]
example_inputs_ids = master_status_j["example_inputs_ids"]
train_assignment = master_status_j["train_assignment"]

llama2_infer.construct_full_cluster_simulator(master_status_j)
llama2_infer.simulator_advance_time(2000000000)

for _ in range(1):
    output_futures = llama2_infer.run_task(train_task_id, fw_first_params_ids + fw_buffer_ids + example_inputs_ids, train_assignment)
    trained_params, out = output_futures[:len(fw_first_params_ids)], output_futures[len(fw_first_params_ids):]
    for future in out + fw_first_params_ids:
        llama2_infer.drop_future(future)
    fw_first_params_ids = trained_params

j = llama2_infer.get_master_status()

llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)
llama2_infer.simulator_advance_time(2000000000)

j = llama2_infer.get_master_status()
llama2_infer.clear_simulator()