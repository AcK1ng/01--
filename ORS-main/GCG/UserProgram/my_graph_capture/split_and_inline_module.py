#!/bin/env python
import torch
from typing import Optional, Dict

def split_module_by_call_function(gm: torch.fx.GraphModule, fun, submod_nr, prefix: str) -> torch.fx.GraphModule:
    if fun == None:
        return gm
    split_point = [node for node in gm.graph.nodes if node.op == "call_function" and node.target == fun]
    if submod_nr < len(split_point):
        n = len(split_point)
        m = submod_nr
    
        if 0 == n % m:
            step = n // m
        else:
            step = (n // m) + 1
    
        split_point = set([split_point[i] for i in range(0, n, step)])
    else:
        split_point = set(split_point)

    partition_counter = 0
    def split_policy(node: torch.fx.Node) -> int:
        nonlocal partition_counter
        nonlocal split_point
        ret = partition_counter
        if node in split_point:
            partition_counter += 1
        return prefix + str(ret)

    return torch.fx.passes.split_module.split_module(gm, None, split_policy)

def split_module_by_modname_prefix(gm: torch.fx.GraphModule, prefix_list: list) -> torch.fx.GraphModule:
    def gen_split_policy(prefix_list: list):
        prefix_now: Optional[str] = None if len(prefix_list) == 0 else prefix_list.pop(0)
        partition_counter = 0
        def inner(node: torch.fx.Node) -> int:
            nonlocal prefix_list
            nonlocal prefix_now
            nonlocal partition_counter
            if node.op == "call_module":
                module_name = node.target
                if prefix_now is None:
                    return partition_counter
                elif module_name.startswith(prefix_now):
                    return partition_counter
                else: # not module_name.startswith(prefix_now)
                    prefix_now = None if len(prefix_list) == 0 else prefix_list.pop(0)
                    partition_counter = partition_counter + 1
                    return partition_counter
            else:
                return partition_counter
        return inner

    return torch.fx.passes.split_module.split_module(gm, None, gen_split_policy(prefix_list))


def __inline_module(gm: torch.fx.GraphModule, inline_mod_name: str, call_mod_node_to_replace: torch.fx.Node):
    # Fetch the inner graph module that we want to inline inside `gm`.
    inline_mod = dict(gm.named_modules())[inline_mod_name]
    assert isinstance(inline_mod, torch.fx.GraphModule)
    # Now actually do the swap. Note that we have to keep track of new nodes that are
    # copied into `gm` -- we do this via replacement_mapping.
    call_mod_args = call_mod_node_to_replace.args
    replacement_mapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    ph_count = 0
    def replacement_fn(node):
        new_node = replacement_mapping[node]
        new_node.meta = node.meta.copy()
        return new_node
    for inline_node in inline_mod.graph.nodes:
        if inline_node.op == "placeholder":
            replacement_mapping[inline_node] = call_mod_args[ph_count]
            ph_count += 1
        elif inline_node.op == "output":
            outputs = inline_node.args[0]
            output_replacements = torch.fx.node.map_arg(outputs, replacement_fn)
            call_mod_node_to_replace.replace_all_uses_with(output_replacements)
        else:
            with gm.graph.inserting_before(call_mod_node_to_replace):
                new_node = gm.graph.node_copy(inline_node, replacement_fn)
            if inline_node.op == "call_module" or inline_node.op == "get_attr":
                orig_target_name = inline_node.target
                new_target_name = inline_mod_name + "_" + orig_target_name
                new_node.target = new_target_name
                assert not hasattr(gm, new_target_name)
                setattr(gm, new_target_name, getattr(inline_mod, orig_target_name))
            replacement_mapping[inline_node] = new_node


def inline_submodule(gm: torch.fx.GraphModule, module_name: str):
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == module_name:
            __inline_module(gm, module_name, node)
    gm.graph.eliminate_dead_code()
    gm.delete_submodule(module_name)


def inline_all_submodule(gm: torch.fx.GraphModule):
    for module_name in gm._modules:
        for node in gm.graph.nodes:
            if node.op == "call_module" and node.target == module_name:
                __inline_module(gm, module_name, node)
    gm.delete_all_unused_submodules()
    gm.graph.eliminate_dead_code()

def inline_submod_of_submod(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for model_name in gm._modules:
        submod = getattr(gm, model_name)
        inline_all_submodule(submod)
    return gm
