#!/bin/env python
import torch
from typing import List, NamedTuple, Tuple
from functorch.compile import aot_module, make_boxed_func
import operator
from . import convert_GraphModule_to_TorchScript
import contextlib
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from torch.export.graph_signature import InputKind
    
from torch._functorch._aot_autograd.descriptors import (
    AOTInput,
    AOTOutput,
    BufferAOTInput,
    DifferentiableAOTInput,
    DifferentiableAOTOutput,
    GradAOTOutput,
    ParamAOTInput,
    PlainAOTInput,
    PlainAOTOutput,
    SubclassGetAttrAOTInput,
    SubclassGetAttrAOTOutput,
    TangentAOTInput,
)

if __name__ == "__main__":
    pass
else:
    from . import split_and_inline_module

def capture_fw_graph(model: torch.nn.Module, orig_args, orig_kwargs) -> torch.fx.GraphModule:
    pass

def capture_fw_bw_graph(model: torch.nn.Module, orig_args) -> torch.fx.GraphModule:
    
    torch.compiler._is_compiling_flag = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True


    import contextlib
    from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
    stack = contextlib.ExitStack()
    
    with stack:
        jwd = aot_export_joint_with_descriptors(stack, model, args = orig_args)

    fw_bw_gm = jwd.graph_module

    from torch._functorch._aot_autograd.fx_utils import get_all_input_and_grad_nodes, get_all_output_and_tangent_nodes
    input_nodes = get_all_input_and_grad_nodes(fw_bw_gm.graph)
    output_nodes = get_all_output_and_tangent_nodes(fw_bw_gm.graph)

    nr_param = len([input for input in input_nodes if isinstance(input, ParamAOTInput)])
    nr_buf = len([input for input in input_nodes if isinstance(input, BufferAOTInput)])
    nr_orig_inputs = len([input for input in input_nodes if isinstance(input, PlainAOTInput)])
    nr_tangents = len([input for input in input_nodes if isinstance(input, TangentAOTInput)])
    nr_orig_output = len([output for output in output_nodes if isinstance(output, PlainAOTOutput)])

    nodes_to_output = fw_bw_gm.graph.output_node().args[0]

    old_placeholders = [node for node in fw_bw_gm.graph.nodes if node.op == "placeholder"]
    new_tangent_placeholders = []
    for i in range(nr_tangents):
        with fw_bw_gm.graph.inserting_before(old_placeholders[-nr_tangents-nr_orig_inputs]):
            placeholder = fw_bw_gm.graph.placeholder(name = "tangent" + str(i))
            new_tangent_placeholders.append(placeholder)

    assert 1 == len(new_tangent_placeholders)

    for new_tangent, old_tangent in zip(new_tangent_placeholders, old_placeholders[-nr_tangents:]):
        old_tangent.replace_all_uses_with(new_tangent)
        fw_bw_gm.graph.erase_node(old_tangent)

    # 把forward-backward graph的输出调成先grad，再普通output的样子
    nodes_to_output = nodes_to_output[nr_orig_output:nr_orig_output + nr_param] + nodes_to_output[:nr_orig_output]

    fw_bw_gm.graph.erase_node(fw_bw_gm.graph.output_node())
    fw_bw_gm.graph.output(nodes_to_output)
    fw_bw_gm.graph.eliminate_dead_code()


    fw_bw_gm.recompile()

    graph_model_params = jwd._aot_state.flat_args[0: nr_param]
    graph_model_bufs = jwd._aot_state.flat_args[nr_param: nr_param + nr_buf]
    tangent = torch.tensor(1.)
    graph_model_bufs.append(tangent)


    return fw_bw_gm, tuple(graph_model_params), tuple(graph_model_bufs), nr_orig_output

def get_fw__from_bw_fw(fw_bw_gm, nr_orig_output):
    graph_t = torch.fx.Graph()
    graph_t.output(graph_t.graph_copy(fw_bw_gm.graph, {}))
    fw_gm = torch.fx.GraphModule({}, graph_t)
    nodes_to_output = fw_gm.graph.output_node().args[0][-nr_orig_output:]
    fw_gm.graph.erase_node(fw_gm.graph.output_node())
    fw_gm.graph.output(nodes_to_output)
    fw_gm.graph.eliminate_dead_code()
    for node in fw_gm.graph.nodes:
        if not node.op == "placeholder":
            continue
        if isinstance(node.meta["desc"], TangentAOTInput):
            fw_gm.graph.erase_node(node)
    fw_gm.recompile()
    
    return fw_gm

def split_fw__parameter_partial(parameter_partial_set,
                                fw_gm) -> torch.fx.GraphModule:
    
    node__to__partial = {}

    for node in fw_gm.graph.nodes:
        if node.op == "call_function":
            all_using, _ = torch.utils._pytree.tree_flatten(node.args)
            all_names = [using.name for using in all_using if isinstance(using, torch.fx.Node)]
            for name in all_names:
                for set_idx, parameter_paritial in enumerate(parameter_partial_set):
                    for param_name_prefix in parameter_paritial:
                        if name.startswith(param_name_prefix):
                            node__to__partial[node] = set_idx
                            break
                        if node in node__to__partial:
                            break


    for node in fw_gm.graph.nodes:
        if node.op == "call_function":
            if node not in node__to__partial:
                node__to__partial[node] = "initial_partial"
            break

    cur_partial = None
    for node in fw_gm.graph.nodes:
        if node.op != "call_function":
            continue

        if node in node__to__partial:
            cur_partial = node__to__partial[node]
        else:
            node__to__partial[node] = cur_partial

    def split_policy(node: torch.fx.Node) -> int:
        return str(node__to__partial[node])
    
    fw_gm = torch.fx.passes.split_module.split_module(fw_gm, None, split_policy)
    fw_gm.recompile()
    
    return fw_gm

def split_fw__and__bw(fun_to_split_fw,
                      fw_submod_nr,
                      fw_output_num,
                      fw_bw_gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    
    if fun_to_split_fw is None:
        return fw_bw_gm

    ####################################################################
    # For forward-backward graph
    fw_outputs = set(fw_bw_gm.graph.output_node().args[0][-fw_output_num:])
    fw_bw_split_point = None # 这个把fw-bw graph中的fw部分和bw部分分割开
    for node in fw_bw_gm.graph.nodes:
        if node in fw_outputs:
            fw_bw_split_point = node

    fw_split_point = []
    
    for node in fw_bw_gm.graph.nodes:
        if node is fw_bw_split_point:
            break

        if node.op == "call_function" and node.target == fun_to_split_fw:
            fw_split_point.append(node)
        
    if fw_submod_nr < len(fw_split_point):
        n = len(fw_split_point)
        m = fw_submod_nr
    
        if 0 == n % m:
            step = n // m
        else:
            step = (n // m) + 1
    
        fw_split_point = set([fw_split_point[i] for i in range(0, n, step)])
    else:
        fw_split_point = set(fw_split_point)



    fw_submod_name__to_bw_submod_name = {}
    topo_order_of_submod = {}
    fw_node__to__fw_submod_name = {}
    bw_node__to__bw_submod_name = {}
    fw_now = 1
    partition_counter = 0
    def split_policy(node: torch.fx.Node) -> int:
        nonlocal fw_now
        nonlocal fw_submod_name__to_bw_submod_name
        nonlocal topo_order_of_submod
        if fw_now == 1:
                
            nonlocal partition_counter
            nonlocal fw_split_point
            nonlocal fw_node__to__fw_submod_name

            fw_submod_name_prefix = "fw"
            bw_submod_name_prefix = "bw"

            fw_submod_name = fw_submod_name_prefix + str(partition_counter)
            bw_submod_name = bw_submod_name_prefix + str(partition_counter)

            if fw_submod_name not in fw_submod_name__to_bw_submod_name:
                fw_submod_name__to_bw_submod_name[fw_submod_name] = bw_submod_name

            if node in fw_split_point:
                partition_counter += 1

            fw_node__to__fw_submod_name[node] = fw_submod_name

            if node == fw_bw_split_point:
                fw_now = 0

                # 马上要开始backward的分划了，在这里求出来每个部分的拓扑序
                topo_order = 0
                for fw in fw_submod_name__to_bw_submod_name.keys():
                    topo_order_of_submod[fw] = topo_order
                    topo_order += 1
                for bw in reversed(fw_submod_name__to_bw_submod_name.values()):
                    topo_order_of_submod[bw] = topo_order
                    topo_order += 1

            return fw_submod_name
        else:
            nonlocal bw_node__to__bw_submod_name
            activations_from = [fw_node__to__fw_submod_name[input] for input in node.args
                                if input in fw_node__to__fw_submod_name]
            inputs_from_bw = [bw_node__to__bw_submod_name[input] for input in node.args
                              if input in bw_node__to__bw_submod_name]
            if 0 < len(activations_from):
                bw_submod_name = fw_submod_name__to_bw_submod_name[activations_from[0]]
            elif 0 < len(inputs_from_bw):
                max_order_bw = None
                max_order = -1
                for bw in inputs_from_bw:
                    topo_order = topo_order_of_submod[bw]
                    if max_order < topo_order:
                        max_order_bw = bw
                bw_submod_name = max_order_bw
            else:
                bw_submod_name = next(iter(fw_submod_name__to_bw_submod_name.values()))
            bw_node__to__bw_submod_name[node] = bw_submod_name
            return bw_submod_name

    fw_bw_gm = torch.fx.passes.split_module.split_module(fw_bw_gm, None, split_policy)
    fw_bw_gm.recompile()

    return fw_bw_gm


def integrate_into_one_big_op(fw_gm):
    def split_policy(node: torch.fx.Node) -> int:
        return "op"
    
    fw_gm = torch.fx.passes.split_module.split_module(fw_gm, None, split_policy)
    fw_gm.recompile()
    
    return fw_gm


if __name__ == "__main__":
    pass





