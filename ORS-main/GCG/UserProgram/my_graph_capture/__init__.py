#!/bin/env python
import torch
import os
import pickle
import shutil
from typing import Tuple, Union
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch._guards import detect_fake_mode

if __name__ == "__main__":
    pass
else:
    from . import capture_graph
    from . import split_and_inline_module
    from . import functionalize_GraphModule
    from . import convert_GraphModule_to_TorchScript
    from . import functionalize_TorchScript
    from . import eliminate_TS_redundant
    from . import optimize_TS
    from . import eliminate_gm_redundant_node
    from . import eliminate_TS_to_be_run
    from . import eliminate_GM_to_be_run
    from . import get_fw_graph__from_grad_graph
    from . import add_optimizer
    
    from .native_executor import *
    from .graph_export import *

def states_to_device_as_example_inputs(states, device):
    flatted, tree_spec = torch.utils._pytree.tree_flatten(states)
    device_placed_flatted = [t.to(device) if isinstance(t, torch.Tensor) else t for t in flatted]
    inputs_t = torch.utils._pytree.tree_unflatten(device_placed_flatted, tree_spec)
    return inputs_t
    
def capture_from_torch_export__stage1(ep, parameter_partial_set = None):
    fw_gm = ep.graph_module

    fake_mode = detect_fake_mode()
    assert fake_mode is not None

    example_inputs = ep.example_inputs[0]
    
    nr_example_inputs = len(example_inputs)


    def get_fake_tensor(real_tensors):
        flatted, tree_spec = pytree.tree_flatten(real_tensors)
        fake_flatted = []
        for t in flatted:
            if isinstance(t, torch.Tensor):
                fake_t = fake_mode.from_tensor(t)
                fake_flatted.append(fake_t)
            else:
                fake_flatted.append(t)

        fake_tensors = pytree.tree_unflatten(fake_flatted, tree_spec)
        return fake_tensors




    model_params = tuple(ep.parameters())
    model_bufs = tuple(ep.buffers()) + tuple(ep.constants.values())

    device = model_params[0].device
    torch.set_default_device(device)

    # model_params = states_to_device_as_example_inputs(model_params, device)
    # model_bufs = states_to_device_as_example_inputs(model_bufs, device)
    example_inputs = states_to_device_as_example_inputs(example_inputs, device)

    fake_model_params = get_fake_tensor(model_params)
    fake_model_bufs = get_fake_tensor(model_bufs)

    orig_module_str = str(fw_gm.graph)

    fw_gm = eliminate_GM_to_be_run.eliminate_to_device(fw_gm)
    fw_gm = eliminate_GM_to_be_run.change_view_to_reshape(fw_gm)
    fw_gm = eliminate_GM_to_be_run.eliminate__device_in_kwargs(fw_gm)
    fw_gm = eliminate_GM_to_be_run.add_device_kwargs__for__some_node(fw_gm, device)
    fw_gm = eliminate_gm_redundant_node.eliminate_redundant_detach(fw_gm)
    fw_gm = eliminate_gm_redundant_node.eliminate_assert(fw_gm)
    if parameter_partial_set is not None:
        fw_gm = capture_graph.split_fw__parameter_partial(parameter_partial_set, fw_gm)
    else:
        fw_gm = capture_graph.integrate_into_one_big_op(fw_gm)
    fw_gm = eliminate_gm_redundant_node.eliminate_redundant_getitem(fw_gm)

    os.environ["PYTORCH_JIT_TYPE_VERBOSITY"] = "4" # TypeVerbosity::Symbolic，把tensor类型中的符号计算用的东西打印出来

    ts_module, ts_submods, symbol__to__symexpr = convert_GraphModule_to_TorchScript.GraphModule_to_TorchScript(fw_gm, *(model_params + model_bufs + example_inputs))


    return ts_module, ts_submods, symbol__to__symexpr, orig_module_str, model_params, model_bufs, example_inputs, nr_example_inputs

def capture_from_torch_export__stage2(_):
    ts_module, ts_submods, symbol__to__symexpr, orig_module_str, model_params, model_bufs, example_input, nr_example_inputs = _

    fake_mode = detect_fake_mode()
    assert fake_mode is None


    ts_module = torch.parse_ir(ts_module.__repr__())
    for submod_name in ts_submods:
        ts_submod = ts_submods[submod_name]
        ts_submod = eliminate_TS_to_be_run.eliminate_wrong_type(ts_submod)
        ts_submods[submod_name] = ts_submod

    ts_module, ts_submods, model_params = eliminate_TS_redundant.merge_root_params(ts_module,
                                                                                   ts_submods,
                                                                                   model_params,
                                                                                   0,
                                                                                   len(model_bufs) + nr_example_inputs,
                                                                                   False)
    # 因为buf一般要序列化之后上传，如果合起来，没办法upload_tensor tuple
    # if 1 < len(model_bufs):
    #     ts_module, ts_submods, model_bufs = eliminate_TS_redundant.merge_root_params(ts_module,
    #                                                                                  ts_submods,
    #                                                                                  model_bufs,
    #                                                                                  len(model_params),
    #                                                                                  nr_example_inputs,
    #                                                                                  False)

    ts_fw_root, ts_fw_submods = ts_module, ts_submods

    def go_optimizing(root, submods):
        optimize_TS.dce(root)
        # torch._C._jit_pass_dce(root)
        root, submods = eliminate_TS_redundant.remove_unused_submods(root, submods)
        root, submods = eliminate_TS_redundant.set_unsed_submod_output__is__None(root, submods)

        root, submods = eliminate_TS_redundant.reduce_tupleIndex(root, submods)
        root, submods = eliminate_TS_redundant.merge_TupleConstruct_for_root_out(root, submods)

        optimize_TS.dce(root)
        # torch._C._jit_pass_dce(root)
        root, submods = eliminate_TS_redundant.set_unsed_submod_output__is__None(root, submods)
        root, submods = eliminate_TS_redundant.eliminate_redundant_NoneType(root, submods)
        root, submods = eliminate_TS_redundant.eliminate_useless_TupleType(root, submods)

        optimize_TS.dce(root)
        # torch._C._jit_pass_dce(root)
        root = eliminate_TS_redundant.merge_constant__for_my_root(root)
        root = eliminate_TS_to_be_run.replace_device(root)
        root, submods = eliminate_TS_redundant.eliminate_useless_placeholder_of_submods(root, submods)

        optimize_TS.constant_propagation(root)
        optimize_TS.dce(root)
        # torch._C._jit_pass_dce(root)
        
        for submod_name in submods:
            submod = submods[submod_name]
            submod = optimize_TS._optimize_graph(submod)
            submod = eliminate_TS_to_be_run.replace_device(submod)
            submod = eliminate_TS_to_be_run.fix_type_for_some_op__to_run(submod)
            optimize_TS.dce(submod)
            # torch._C._jit_pass_dce(submod)
            submods[submod_name] = submod

        return root, submods
    
    ts_fw_root, ts_fw_submods = go_optimizing(ts_fw_root, ts_fw_submods)

    def go_serialize(root, submods):
        serialized_submods = {}
        for submod_name, submod in submods.items():
            serialized_submods[submod_name] = submod.__repr__()
        return root.__repr__(), serialized_submods

    ts_fw_root, ts_fw_submods = go_serialize(ts_fw_root, ts_fw_submods)

    graph_pack = {}
    graph_pack["orig_module_str"] = orig_module_str
    graph_pack["ts_fw_root"] = ts_fw_root
    graph_pack["ts_fw_submods"] = ts_fw_submods
    graph_pack["fw_params"] = model_params
    graph_pack["fw_buffer"] = model_bufs
    graph_pack["symbol__to__symexpr"] = symbol__to__symexpr
    graph_pack["ts_grad_root"] = "graph():\n return ()"
    graph_pack["ts_grad_submods"] = {}
    return graph_pack







def capture_training_graph_stage_1(module: torch.nn.Module, fun_to_split_fw, fw_submod_nr, optimizer, args):

    example_inputs, _ = pytree.tree_flatten(args)
    example_inputs = tuple(example_inputs)
    device = example_inputs[0].device


    nr_example_inputs = len(example_inputs)

    fake_mode = detect_fake_mode()
    assert fake_mode is not None

    # TorchScript和torch.fx最大的区别在于， TorchScript中有类型信息！！如果类型不对，jit是会报错的！！
    # 如果要新加结点，切记切记考虑类型，考虑写TorchScript的pass还是torch.fx的pass
    fw_bw_gm, model_params, model_bufs, fw_output_num = capture_graph.capture_fw_bw_graph(module, args)
    model_params = states_to_device_as_example_inputs(model_params, device)
    model_bufs = states_to_device_as_example_inputs(model_bufs, device)

    fake_param__to__real_param = dict(zip(model_params, module.parameters()))
    fake_buf__to__real_buf = dict(zip(model_bufs, module.buffers()))

    flatted, spec = pytree.tree_flatten(model_params)
    new_flatted = []
    for param in flatted:
        new_flatted.append(fake_param__to__real_param[param])
    model_params = pytree.tree_unflatten(new_flatted, spec)
    
    flatted, spec = pytree.tree_flatten(model_bufs)
    new_flatted = []
    for buf in flatted:
        if buf in fake_buf__to__real_buf:
            t = fake_buf__to__real_buf[buf]
        else:
            t = buf.constant
        new_flatted.append(t)
    model_bufs = pytree.tree_unflatten(new_flatted, spec)

    if optimizer == None:
        pass
    elif optimizer == "SGD":
        fw_bw_gm = add_optimizer.add_SGD(fw_output_num, fw_bw_gm, 0.01)
    else:
        assert False

    orig_module_str = str(fw_bw_gm.graph)

    fw_bw_gm = eliminate_GM_to_be_run.eliminate_to_device(fw_bw_gm)
    fw_bw_gm = eliminate_GM_to_be_run.change_view_to_reshape(fw_bw_gm)
    fw_bw_gm = eliminate_GM_to_be_run.eliminate__device_in_kwargs(fw_bw_gm)
    fw_bw_gm = eliminate_GM_to_be_run.add_device_kwargs__for__some_node(fw_bw_gm, device)
    fw_bw_gm = eliminate_gm_redundant_node.eliminate_redundant_detach(fw_bw_gm)
    fw_bw_gm = eliminate_gm_redundant_node.eliminate_assert(fw_bw_gm)
    fw_bw_gm = capture_graph.split_fw__and__bw(fun_to_split_fw,
                                               fw_submod_nr,
                                               fw_output_num,
                                               fw_bw_gm)

    fw_bw_gm = eliminate_gm_redundant_node.eliminate_redundant_getitem(fw_bw_gm)

    os.environ["PYTORCH_JIT_TYPE_VERBOSITY"] = "4" # TypeVerbosity::Symbolic，把tensor类型中的符号计算用的东西打印出来

    ts_module, ts_submods, symbol__to__symexpr = convert_GraphModule_to_TorchScript.GraphModule_to_TorchScript(fw_bw_gm, *(model_params + model_bufs + example_inputs))

    return (nr_example_inputs, model_params, model_bufs, orig_module_str, ts_module, ts_submods, fw_output_num, device, symbol__to__symexpr)

def capture_training_graph_stage_2(_):
    nr_example_inputs, model_params, model_bufs, orig_module_str, ts_module, ts_submods, fw_output_num, device, symbol__to__symexpr = _

    fake_mode = detect_fake_mode()
    assert fake_mode is None


    ts_module = torch.parse_ir(ts_module.__repr__())
    for submod_name in ts_submods:
        ts_submod = ts_submods[submod_name]
        ts_submod = eliminate_TS_to_be_run.eliminate_wrong_type(ts_submod)
        ts_submod = optimize_TS._optimize_graph(ts_submod)
        ts_submods[submod_name] = ts_submod

    ts_module, ts_submods, model_params = eliminate_TS_redundant.merge_root_params(ts_module,
                                                                                   ts_submods,
                                                                                   model_params,
                                                                                   0,
                                                                                   len(model_bufs) + nr_example_inputs,
                                                                                   True)
    # if 1 < len(model_bufs):
    #     ts_module, ts_submods, model_bufs = eliminate_TS_redundant.merge_root_params(ts_module,
    #                                                                                  ts_submods,
    #                                                                                  model_bufs,
    #                                                                                  len(model_params),
    #                                                                                  nr_example_inputs,
    #                                                                                  False)

    ts_grad_root, ts_grad_submods = ts_module, ts_submods
    ts_fw_root, ts_fw_submods = get_fw_graph__from_grad_graph.get_fw_graph__from_grad_graph(ts_grad_root, ts_grad_submods, fw_output_num)

    def go_optimizing(root, submods):
        torch._C._jit_pass_dce(root)
        root, submods = eliminate_TS_redundant.remove_unused_submods(root, submods)
        root, submods = eliminate_TS_redundant.set_unsed_submod_output__is__None(root, submods)

        root, submods = eliminate_TS_redundant.reduce_tupleIndex(root, submods)
        root, submods = eliminate_TS_redundant.merge_TupleConstruct_for_root_out(root, submods)

        torch._C._jit_pass_dce(root)
        root, submods = eliminate_TS_redundant.set_unsed_submod_output__is__None(root, submods)
        root, submods = eliminate_TS_redundant.eliminate_redundant_NoneType(root, submods)
        root, submods = eliminate_TS_redundant.eliminate_useless_TupleType(root, submods)

        torch._C._jit_pass_dce(root)
        root = eliminate_TS_redundant.merge_constant__for_my_root(root)
        root = eliminate_TS_to_be_run.replace_device(root)
        root, submods = eliminate_TS_redundant.eliminate_useless_placeholder_of_submods(root, submods)

        torch._C._jit_pass_dce(root)
    
        for submod_name in submods:
            submod = submods[submod_name]
            submod = optimize_TS._optimize_graph(submod)
            submod = eliminate_TS_to_be_run.replace_device(submod)
            submod = eliminate_TS_to_be_run.fix_type_for_some_op__to_run(submod)
            torch._C._jit_pass_dce(submod)
            submods[submod_name] = submod

        return root, submods
    
    ts_grad_root, ts_grad_submods = go_optimizing(ts_grad_root, ts_grad_submods)
    ts_fw_root, ts_fw_submods = go_optimizing(ts_fw_root, ts_fw_submods)

    def go_serialize(root, submods):
        serialized_submods = {}
        for submod_name, submod in submods.items():
            serialized_submods[submod_name] = submod.__repr__()
        return root.__repr__(), serialized_submods

    ts_grad_root, ts_grad_submods = go_serialize(ts_grad_root, ts_grad_submods)
    ts_fw_root, ts_fw_submods = go_serialize(ts_fw_root, ts_fw_submods)

    graph_pack = {}
    graph_pack["orig_module_str"] = orig_module_str
    graph_pack["ts_fw_root"] = ts_fw_root
    graph_pack["ts_fw_submods"] = ts_fw_submods
    graph_pack["ts_grad_root"] = ts_grad_root
    graph_pack["ts_grad_submods"] = ts_grad_submods
    graph_pack["fw_params"] = model_params
    graph_pack["fw_buffer"] = model_bufs
    graph_pack["symbol__to__symexpr"] = symbol__to__symexpr
    return graph_pack


