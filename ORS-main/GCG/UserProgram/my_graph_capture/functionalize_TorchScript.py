#!/bin/env python
import torch
from typing import Tuple, Dict

def __functionalize_TS(mod: torch.Graph) -> Tuple[torch.Graph, Tuple]:
    extra_states = []
    nodes_to_delete = []
    for node in mod.nodes():
        extra_state = None
        if node.kind() == "prim::Constant":
            if node.hasAttribute("value") and node.kindOf("value") == "t":
                extra_state = node.t("value")
        
        if extra_state is None:
            continue
        orig_value = node.outputsAt(0)
        new_value = mod.addInput("constant_tensor")
        orig_value.replaceAllUsesWith(new_value)
        nodes_to_delete.append(node)
        extra_states.append(extra_state)

    for node in nodes_to_delete:
        node.destroy()

    return mod, tuple(extra_states)


def extra_const_tensor_after_TS_optimize(root: torch.Graph, submods: Dict, preserve_prefix_placeholder):
    cur_states = []
    all_call_graph_node = [node for node in root.nodes() if node.kind() == "prim::GCG_Call_Submod"]
    for submod_name in submods:
        submod = submods[submod_name]
        submod, extra_states = __functionalize_TS(submod)
        submods[submod_name] = submod

        extra_state_num = len(extra_states)

        if extra_state_num == 0:
            continue

        extra_state_values = [root.addInput("const_tensor") for extra_state in extra_states]
        all_input_num = len(list(root.inputs()))

        # Placeholders before permuting
        #                                                       |<- extra_state_num ->|
        # -----------------------------------------------------------------------------
        # | 1. Don't change! | 2. cur_states | 3. Don't change! |  4. extra_states    |
        # -----------------------------------------------------------------------------
        #                     ^                                  ^                     ^
        #          preserve_prefix_placeholder                                   all_input_num
        # 1. list(range(0, preserve_prefix_placeholder))
        # (2 + 3). list(range(preserve_prefix_placeholder, all_input_num - extra_state_num))
        # 4. list(range(all_input_num - extra_state_num, all_input_num))

        perlist = list(range(0, preserve_prefix_placeholder)) + \
                  list(range(all_input_num - extra_state_num, all_input_num)) + \
                  list(range(preserve_prefix_placeholder, all_input_num - extra_state_num))
        root.permuteInputs(perlist)

        # Placeholders after permuting
        #                    |<- extra_state_num ->|
        # -----------------------------------------------------------------------------
        # | 1. Don't change! |  4. extra_states    | 2. cur_states | 3. Don't change! |
        # -----------------------------------------------------------------------------
        #                     ^                                                        ^
        #          preserve_prefix_placeholder                                    all_input_num

        cur_states = list(extra_states) + cur_states
        
        for node in [node for node in all_call_graph_node if node.s("target") == submod_name]:
            [node.addInput(extra_state_value) for extra_state_value in extra_state_values]
    return root, submods, tuple(cur_states)