import torch
import os
import pickle
import shutil
import io
from typing import Tuple, Union


def tensors__to__tensors_descriptors(tensors):
    def tensor__to__tensor_descriptor(t):
        assert isinstance(t, torch.Tensor)
        return (t.shape, t.dtype, t.layout)
    if isinstance(tensors, torch.Tensor):
        return tensor__to__tensor_descriptor(tensors)
    elif isinstance(tensors, tuple):
        return [tensors__to__tensors_descriptors(t) for t in tensors]
    else:
        assert False
        

def tensors_descriptors__to__empty_tensors(tensors_descriptors, device):
    def is__tensor_descriptor(t):
        if isinstance(t, tuple) and len(t) == 3 and isinstance(t[1], torch.dtype):
            return True
        return False
    def tensor_descriptor__to__empty_tensor(t):
        shape, dtype, layout = t
        return torch.ops.aten.zeros.default(shape, dtype = dtype, layout = layout, device = device)
    if is__tensor_descriptor(tensors_descriptors):
        return tensor_descriptor__to__empty_tensor(tensors_descriptors)
    elif isinstance(tensors_descriptors, list):
        return tuple([tensors_descriptors__to__empty_tensors(t, device) for t in tensors_descriptors])
    else:
        assert False

def tensors_descriptors__to_TS_graph(tensors_descriptors):

    def tensors_descriptors__to_TS_graph_1(tensors_descriptors):
        def is__tensor_descriptor(t):
            if isinstance(t, tuple) and len(t) == 3 and isinstance(t[1], torch.dtype):
                return True
            return False
        def tensor_descriptor__to__empty_tensor_graph(t):
            def scalar_type_to_int(scalar_type):
                return torch._C._jit_interpret_graph(torch.parse_ir("""
                  graph(%8):
                    return (%8)
                """), (scalar_type, ))

            def torchSize__to__arrstr(torchSize):
                return torchSize.__str__()[11:-1]

            shape, dtype, layout = t

            # 这里还是用zeros而不是empty。因为empty可能包含未初始化的数据，导致后续的graph执行出错（比如embedding）
            empty_tensor__template = """
graph():
  %shape : int[] = prim::Constant[value={shape}]()
  %dtype : ScalarType = prim::Constant[value={dtype}]()
  %layout : Layout = prim::Constant[value={layout}]()
  %1 : NoneType = prim::Constant()
  %device: Device = prim::GCG_get_native_device()
  %empty : Tensor = aten::zeros(%shape, %dtype, %layout, %device, %1)
  return (%empty)
"""

            empty_tensor_constant__template = """
graph():
  %shape : int[] = prim::ListConstruct()
  %dtype : ScalarType = prim::Constant[value={dtype}]()
  %layout : Layout = prim::Constant[value={layout}]()
  %1 : NoneType = prim::Constant()
  %device: Device = prim::GCG_get_native_device()
  %empty : Tensor = aten::zeros(%shape, %dtype, %layout, %device, %1)
  return (%empty)
"""
            if torchSize__to__arrstr(shape) == "[]":
                empty_tensor = empty_tensor_constant__template.format(shape = torchSize__to__arrstr(shape),
                                                                      dtype = scalar_type_to_int(dtype),
                                                                      layout = scalar_type_to_int(layout))
            else:
                empty_tensor = empty_tensor__template.format(shape = torchSize__to__arrstr(shape),
                                                             dtype = scalar_type_to_int(dtype),
                                                             layout = scalar_type_to_int(layout))
            ts = torch.parse_ir(empty_tensor)
            next(iter(ts.outputs())).setType(torch.TensorType.get().with_dtype(dtype).with_sizes(shape))
            return ts

        if is__tensor_descriptor(tensors_descriptors):
            return tensor_descriptor__to__empty_tensor_graph(tensors_descriptors)
        elif isinstance(tensors_descriptors, list):
            graph = torch.Graph()
            sub_graphs = [tensors_descriptors__to_TS_graph_1(t) for t in tensors_descriptors]
            values_in_graph = [(graph.insertGraph(one_of_subgraphs, []))[0] for one_of_subgraphs in sub_graphs]
            tuple_constructor_node = graph.create("prim::TupleConstruct", values_in_graph, 1)
            graph.insertNode(tuple_constructor_node)
            output_value = tuple_constructor_node.outputsAt(0)
            output_value.setType(torch.TupleType([value.type() for value in values_in_graph]))
            graph.registerOutput(output_value)
            return graph
        else:
            assert False

    assert isinstance(tensors_descriptors, list)

    graph = torch.Graph()
    sub_graphs = [tensors_descriptors__to_TS_graph_1(t) for t in tensors_descriptors]
    sub_graphs__1 = dict()
    for i in range(len(sub_graphs)):
        one_of_subgraphs = sub_graphs[i]
        name = "tensors_" + str(i)
        call_subgraph = graph.create("prim::GCG_Call_Submod", [], 1)
        call_subgraph.s_("target", name)
        call_subgraph.outputsAt(0).setTypeAs(list(one_of_subgraphs.outputs())[0])
        graph.insertNode(call_subgraph)
        graph.registerOutput(call_subgraph.outputsAt(0))
        sub_graphs__1[name] = one_of_subgraphs.__repr__()
        
    return (graph.__repr__(), sub_graphs__1)


def save_to(graph_pack, directory):
    save_without_states_to(graph_pack, directory)
    fw_params = graph_pack["fw_params"]
    torch.save(fw_params, os.path.join(directory, "fw_params.pt"))
    

def save_without_states_to(graph_pack, directory):
    orig_module_str = graph_pack["orig_module_str"]
    ts_fw_root = graph_pack["ts_fw_root"]
    ts_fw_submods = graph_pack["ts_fw_submods"]
    ts_grad_root = graph_pack["ts_grad_root"]
    ts_grad_submods = graph_pack["ts_grad_submods"]
    fw_params = graph_pack["fw_params"]
    fw_buffer = graph_pack["fw_buffer"]
    symbol__to__symexpr = graph_pack["symbol__to__symexpr"]
    example_inputs = graph_pack["example_inputs"]

    shutil.rmtree(directory, ignore_errors = True)
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "orig_GraphModule.txt"), "w") as f:
        f.write(orig_module_str)

    with open(os.path.join(directory, "symbol__to__symexpr.pickle"), "wb") as f:
        pickle.dump(symbol__to__symexpr, f)

    visualize_ts(ts_fw_root, os.path.join(directory, "ts_fw_root.dot"))
    with open(os.path.join(directory, "ts_fw_root.txt"), "w") as f:
        f.write(ts_fw_root)
    submods_list = [name for name in ts_fw_submods]
    with open(os.path.join(directory, "ts_fw_submod_list.pickle"), "wb") as f:
        pickle.dump(submods_list, f)
    for submod_name, ts_submod in ts_fw_submods.items():
        with open(os.path.join(directory, "ts_fw_submods_" + submod_name + ".txt"), "w") as f:
            f.write(ts_submod)


    visualize_ts(ts_grad_root, os.path.join(directory, "ts_grad_root.dot"))
    with open(os.path.join(directory, "ts_grad_root.txt"), "w") as f:
        f.write(ts_grad_root)
    submods_list = [name for name in ts_grad_submods]
    with open(os.path.join(directory, "ts_grad_submod_list.pickle"), "wb") as f:
        pickle.dump(submods_list, f)
    for submod_name, ts_submod in ts_grad_submods.items():
        with open(os.path.join(directory, "ts_grad_submods_" + submod_name + ".txt"), "w") as f:
            f.write(ts_submod)

    with open(os.path.join(directory, "fw_params_shapes.pickle"), "wb") as f:
        pickle.dump(tensors__to__tensors_descriptors(fw_params), f)
    fw_buffer = graph_pack["fw_buffer"]
    torch.save(fw_buffer, os.path.join(directory, "fw_buffer.pt"))

    torch.save(example_inputs, os.path.join(directory, "example_inputs.pt"))

def read_from_no_states(directory: str, device = "cpu"):
    with open(os.path.join(directory, "symbol__to__symexpr.pickle"), "rb") as f:
        symbol__to__symexpr = pickle.load(f)

    with open(os.path.join(directory, "ts_fw_root.txt"), "r") as f:
        ts_fw_root = f.read()
    ts_fw_submods = {}
    with open(os.path.join(directory, "ts_fw_submod_list.pickle"), "rb") as f:
        submods_list = pickle.load(f)
    for submod_name in submods_list:
        with open(os.path.join(directory, "ts_fw_submods_" + submod_name + ".txt"), "r") as f:
            ts_fw_submods[submod_name] = f.read()


    with open(os.path.join(directory, "ts_grad_root.txt"), "r") as f:
        ts_grad_root = f.read()
    ts_grad_submods = {}
    with open(os.path.join(directory, "ts_grad_submod_list.pickle"), "rb") as f:
        submods_list = pickle.load(f)
    for submod_name in submods_list:
        with open(os.path.join(directory, "ts_grad_submods_" + submod_name + ".txt"), "r") as f:
            ts_grad_submods[submod_name] = f.read()

    with open(os.path.join(directory, "fw_params_shapes.pickle"), "rb") as f:
        fw_params_shapes = pickle.load(f)

    fw_buffer = torch.load(os.path.join(directory, "fw_buffer.pt"), map_location=torch.device(device))

    example_inputs = torch.load(os.path.join(directory, "example_inputs.pt"), map_location=torch.device(device))

    graph_pack = {}
    graph_pack["ts_fw_root"] = ts_fw_root
    graph_pack["ts_fw_submods"] = ts_fw_submods
    graph_pack["ts_grad_root"] = ts_grad_root
    graph_pack["ts_grad_submods"] = ts_grad_submods
    graph_pack["fw_params_shapes"] = fw_params_shapes
    graph_pack["fw_buffer"] = fw_buffer
    graph_pack["symbol__to__symexpr"] = symbol__to__symexpr
    graph_pack["example_inputs"] = example_inputs
    return graph_pack

def read_from_dummy_state(directory: str, device = "cpu"):
    graph_pack = read_from_no_states(directory, device)
    return graph_pack

def read_from(directory: str, device = "cpu"):
    graph_pack = read_from_no_states(directory, device)
    fw_params = torch.load(os.path.join(directory, "fw_params.pt"), map_location=torch.device(device))
    graph_pack["fw_params"] = fw_params
    return graph_pack