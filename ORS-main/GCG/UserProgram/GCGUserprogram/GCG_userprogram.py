
import torch
import requests
import json
import io
import os
from typing import TypeAlias, List, Optional, Dict, Callable, Tuple, Any, Set, Union
import GCGUserprogram


TaskID: TypeAlias = int
Future: TypeAlias = int
Rank: TypeAlias = int

class GCGProgramPythonPort:
    def __init__(self, cluster_wrapper):
        self.cluster_wrapper = cluster_wrapper

    def debug(self):
        self.cluster_wrapper.DEBUG()

    def submit_graph(self,
                     root_graph: str,
                     sub_graphs: Dict[str, str],
                     symbol__to__symexpr: Dict[int, str] = {}) -> TaskID:
        return self.cluster_wrapper.SubmitGraph(root_graph, sub_graphs, symbol__to__symexpr)

    def drop_task(self, task_id: TaskID):
        return self.cluster_wrapper.DropTask(task_id)
    
    def run_task(self,
                 task_id: TaskID,
                 inputs: List[Future],
                 manual_assignment: Optional[List[Rank]] = None,
                 debug_output_i: List[int] = []) -> List[Future]:
        return self.cluster_wrapper.RunTask(task_id, inputs, manual_assignment, debug_output_i)

    def drop_future(self, future: Future):
        self.cluster_wrapper.DropFuture(future)

    def new_empty_tensors(self, tensors_descriptors, placement: Optional[Union[List, Rank]] = None) -> List[Future]:
        root_graph, sub_graphs = GCGUserprogram.tensors_descriptors__to_TS_graph(tensors_descriptors)
        task_id = self.submit_graph(root_graph, sub_graphs)
        if isinstance(placement, Rank):
            placement = [placement] * len(tensors_descriptors)
        outputs = self.run_task(task_id, [], placement)
        self.drop_task(task_id)
        return outputs

    def upload_tensor(self, t: torch.Tensor) -> Future:
        assert(torch.is_tensor(t))
        
        def scalar_type_to_int(scalar_type):
            return torch._C._jit_interpret_graph(torch.parse_ir("""
              graph(%8: dtype):
                return (%8)
            """), (scalar_type, ))

        buff = io.BytesIO()
        torch.save(t, buff)

        return self.cluster_wrapper.UploadTensor_ToCluster(buff.getbuffer(), t.shape, scalar_type_to_int(t.dtype))




class GCGProgram:
    def __init__(self, url: str):
        self.url = url

    def debug(self):
        requests.post("http://{url}/DEBUG".format(url = self.url))

    def submit_graph(self,
                     root_graph: str,
                     sub_graphs: Dict[str, str],
                     symbol__to__symexpr: Dict[int, str] = {}) -> TaskID:
        graph_task = {}
        graph_task["graph"] = root_graph
        graph_task["submods"] = sub_graphs
        graph_task["symbol__to__symexpr"] = list(symbol__to__symexpr.items())
        r = requests.post("http://{url}/submitgraph".format(url = self.url),
                          json.dumps(graph_task))
    
        task_id = json.loads(r.text)
        return task_id

    def drop_task(self, task_id: TaskID):
        dict1 = {}
        dict1["task_id"] = task_id
        r = requests.post("http://{url}/droptask".format(url = self.url),
                          json.dumps(dict1))

    def run_task(self,
                 task_id: TaskID,
                 inputs: List[Future],
                 manual_assignment: Optional[List[Rank]] = None,
                 debug_output_i: List[int] = []) -> List[Future]:
        dict1 = {}
        dict1["task_id"] = task_id
        dict1["inputs"] = inputs
        if manual_assignment is not None:
            dict1["manual_assignment"] = manual_assignment
        dict1["debug_output_i"] = debug_output_i
        r = requests.post("http://{url}/runtask".format(url = self.url),
                          json.dumps(dict1))
    
        outputs = json.loads(r.text)
        return outputs
    
    def drop_future(self, future: Future):
        dict1 = {}
        dict1["future"] = future
        r = requests.post("http://{url}/dropfuture".format(url = self.url),
                          json.dumps(dict1))

    def new_empty_tensors(self, tensors_descriptors, placement: Optional[Union[List, Rank]] = None) -> List[Future]:
        root_graph, sub_graphs = GCGUserprogram.tensors_descriptors__to_TS_graph(tensors_descriptors)
        task_id = self.submit_graph(root_graph, sub_graphs)
        if isinstance(placement, Rank):
            placement = [placement] * len(tensors_descriptors)
        outputs = self.run_task(task_id, [], placement)
        self.drop_task(task_id)
        return outputs

    def upload_tensor(self, t: torch.Tensor) -> Future:
        assert(torch.is_tensor(t))
        
        def scalar_type_to_int(scalar_type):
            return torch._C._jit_interpret_graph(torch.parse_ir("""
              graph(%8: dtype):
                return (%8)
            """), (scalar_type, ))

        buff = io.BytesIO()
        torch.save(t, buff)

        dict1 = {}
        dict1["serialized_data"] = list(buff.getvalue())
        dict1["shape"] = t.shape
        dict1["dtype"] = scalar_type_to_int(t.dtype)

        r = requests.post("http://{url}/uploadtensor".format(url = self.url),
                          json.dumps(dict1))
        output = json.loads(r.text)
        return output

    def get_master_status(self):
        r = requests.post("http://{url}/masterstatus".format(url = self.url))
        return json.loads(r.text)
    
    def construct_full_cluster_simulator(self, master_status_j):
        r = requests.post("http://{url}/constructfullclustersimulator".format(url = self.url), json.dumps(master_status_j))

    def simulator_advance_time(self, t):
        r = requests.post("http://{url}/simulatoradvancetime".format(url = self.url), json.dumps(t))

    def clear_simulator(self):
        r = requests.post("http://{url}/clearsimulator".format(url = self.url))



class AzureDataTrace:
    def __init__(self,
                 path: str,
                 callback: Callable,
                 time_scaling: float = 1):
        self.path = path
        self.time_scaling = time_scaling
        self.callback = callback
        self.timers = []


        from datetime import datetime
        import time
        cday = datetime.strptime('2017-8-1 18:20:20', '%Y-%m-%d %H:%M:%S')

        with open(self.path, 'r') as f:
            line_iter = iter(f)
            head_lien = next(line_iter)

            line = next(line_iter)
            row = line.strip().split(',')

            format_str = "%Y-%m-%d %H:%M:%S.%f0"

            first_timestamp = datetime.strptime(row[0], format_str).timestamp()

            next_timestamp = (first_timestamp - first_timestamp) * self.time_scaling
            next_contextlen = row[1]
            next_generated = row[2]

            for line in line_iter:
                from threading import Timer

                self.timers.append(Timer(next_timestamp, callback, args=(next_timestamp, next_contextlen, next_generated)))

                row = line.strip().split(',')
                t = datetime.strptime(row[0], format_str).timestamp()
                next_timestamp = (t - first_timestamp) * self.time_scaling
                next_contextlen = row[1]
                next_generated = row[2]


    def run(self):
        for timer in self.timers:
            timer.start()