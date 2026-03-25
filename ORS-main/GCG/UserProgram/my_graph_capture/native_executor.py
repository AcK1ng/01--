#!/bin/env python
import torch
from typing import TypeAlias, List, Optional, Dict, Callable, Tuple, Any, Set
import pickle
import torch.utils._pytree as pytree

if __name__ == "__main__":
    pass
else:
    from . import graph_export

def native_debug_run(graph_pack, device, bw = True):
    if bw:
        ts_mod, ts_submods = graph_pack["ts_grad_root"], graph_pack["ts_grad_submods"]
    else:
        ts_mod, ts_submods = graph_pack["ts_fw_root"], graph_pack["ts_fw_submods"]

    ts_mod = torch.parse_ir(ts_mod)

    if "fw_params" in graph_pack:
        fw_params = graph_pack["fw_params"]
    else:
        fw_params_shapes = graph_pack["fw_params_shapes"]
        fw_params = graph_export.tensors_descriptors__to__empty_tensors(fw_params_shapes, device)

    fw_buffer = graph_pack["fw_buffer"]

    example_inputs = graph_pack["example_inputs"]

    inputs = (fw_params + fw_buffer + example_inputs)
    flatted, tree_spec = torch.utils._pytree.tree_flatten(inputs)
    flatted = [t.to(device) if isinstance(t, torch.Tensor) else t for t in flatted]
    inputs_t = torch.utils._pytree.tree_unflatten(flatted, tree_spec)
    return execute_TS(ts_mod, ts_submods, inputs_t, device)

def execute_TS(ts_mod: torch.Graph,
               ts_submods: Dict[str, torch.Graph],
               inputs: Tuple[Any],
               device):
    variables: Dict[str, Any] = dict()
    value_use_cnt = {}

    for input_value, input_actuall_parameter in zip(ts_mod.inputs(), inputs):
        input_name = input_value.debugName()
        value_use_cnt[input_name] = len(input_value.uses())
        variables[input_name] = input_actuall_parameter

    for node in ts_mod.nodes():
        actually_param_inputs = [variables[input_value.debugName()]
                                 for input_value in node.inputs()]
        qualified_name = node.kind()
        
        if qualified_name == "prim::GCG_Call_Submod":
            submod = torch.parse_ir(ts_submods[node.s("target")])
            # try:
            res = execute_TS(submod, None, tuple(actually_param_inputs), device)
            # res = torch._C._jit_interpret_graph(submod, tuple(actually_param_inputs))
            # except Exception as e:
            #     flatted, spec = pytree.tree_flatten(actually_param_inputs)
            #     new_flatted = []
            #     for elem in flatted:
            #         t = elem
            #         if isinstance(t, torch.Tensor):
            #             t = t.to("cpu")
            #         new_flatted.append(t)
            #     unflatted = pytree.tree_unflatten(new_flatted, spec)
            #     with open("breakpoint.pickle", "wb") as f:
            #         pickle.dump((submod.__repr__(), tuple(unflatted)), f)
            #     raise e
        elif qualified_name == "prim::GCG_get_native_device":
            res = torch.device(device)
        elif qualified_name == "prim::Constant":
            if node.hasAttribute("value"):
                kind = node.kindOf("value")
                res = getattr(node, kind)("value")
            else:
                res = None
        elif qualified_name == "prim::TupleIndex":
            res = actually_param_inputs[0][actually_param_inputs[1]]
        elif qualified_name == "prim::ListConstruct":
            res = actually_param_inputs
        elif qualified_name == "prim::TupleConstruct":
            res = tuple(actually_param_inputs)
        else:
            names = qualified_name.split("::")
            namespace, opname = names[0], names[1]
            lib = torch.ops.__getattr__(namespace)
            op = lib.__getattr__(opname)
            run_ok = False
            try:
                res = op(*actually_param_inputs)
                run_ok = True
            except Exception as e:
                for overload_name in op._overload_names:
                    overloaded_op = op.__getattr__(overload_name)
                    if len(overloaded_op._schema.arguments) != len(actually_param_inputs):
                        continue

                    t_args = []
                    t_kwargs = {}
                    for arg, param in zip(actually_param_inputs, overloaded_op._schema.arguments):
                        if param.kwarg_only:
                            t_kwargs[param.name] = arg
                        else:
                            t_args.append(arg)
                                
                    try:
                        res = overloaded_op(*tuple(t_args), **t_kwargs)
                        run_ok = True
                        break
                    except:
                        pass
                if not run_ok:
                    print(e)
            assert run_ok
        flatted, tree_spec = torch.utils._pytree.tree_flatten(res)
        contiguoused_flatted = [f.contiguous() if isinstance(f, torch.Tensor) else f for f in flatted]
        res = torch.utils._pytree.tree_unflatten(contiguoused_flatted, tree_spec)

        for input_value in node.inputs():
            input_name = input_value.debugName()
            value_use_cnt[input_name] -= 1
            if value_use_cnt[input_name] == 0:
                del variables[input_name] 

        def register_one_output(output_value, elem):
            output_name = output_value.debugName()
            variables[output_name] = elem
            value_use_cnt[output_name] = len(output_value.uses())

        if 1 == node.outputsSize():
            register_one_output(node.outputsAt(0), res)
        else:
            assert node.outputsSize() == len(res)
            for out, output_value in zip(res, node.outputs()):
                register_one_output(output_value, out)

    output_nr = len(list(ts_mod.outputs()))
    if  1 == output_nr:
        return [variables[output_value.debugName()] for output_value in ts_mod.outputs()][0]
    else:
        return tuple([variables[output_value.debugName()] for output_value in ts_mod.outputs()])
