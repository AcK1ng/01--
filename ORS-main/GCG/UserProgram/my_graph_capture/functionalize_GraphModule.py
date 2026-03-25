#!/bin/env python
import torch
from typing import Tuple

def extract_states_from_GM(gm: torch.fx.GraphModule, preserve_prefix_placeholder):
    states = []
    orig_input_num = len([node for node in gm.graph.nodes if node.op == "placeholder"])

    num_of_placeholder__before_pointed_node = 0
    for node in gm.graph.nodes:
        pointed_node = node
        if num_of_placeholder__before_pointed_node == preserve_prefix_placeholder:
            break


        if node.op == "placeholder":
            num_of_placeholder__before_pointed_node += 1


    for node in gm.graph.nodes:
        if node.op == "get_attr":
            attr_name = node.target
            with gm.graph.inserting_before(pointed_node):
                new_placeholder = gm.graph.placeholder(attr_name)
            node.replace_all_uses_with(new_placeholder)
            states.append(getattr(gm, attr_name))
            delattr(gm, attr_name)
            gm.graph.erase_node(node)

    gm.recompile()

    return gm, tuple(states), orig_input_num

def functionalize_submod(gm: torch.fx.GraphModule, submod_name: str):
    submod = getattr(gm, submod_name)
    submod, states, _ = extract_states_from_GM(submod, 0)

    the_location_to_insert = None
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            the_location_to_insert = node
            break

    state_input_nodes = []
    for i, state in enumerate(states):
        state_name = submod_name + "_state_" + str(i)
        assert not hasattr(gm, state_name)
        setattr(gm, state_name, state)
        with gm.graph.inserting_before(the_location_to_insert):
            state_input_nodes.append(gm.graph.get_attr(state_name))

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == submod_name:
            assert len(node.kwargs) == 0
            node.args = tuple(state_input_nodes) + node.args

    gm.recompile()

def dedupe_states(gm: torch.fx.GraphModule,
                  states: Tuple,
                  preserve_prefix_placeholder,
                  preserve_suffix_placeholder):
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    placeholders = placeholders[preserve_prefix_placeholder:len(placeholders) - preserve_suffix_placeholder]
    assert len(placeholders) == len(states)
    deduped_states_list = []
    deduped_states_dict = dict()
    for state, placeholder in zip(states, placeholders):
        if state not in deduped_states_dict:
            deduped_states_dict[state] = placeholder
            deduped_states_list.append(state)
        else:
            first_placeholder=deduped_states_dict[state]
            placeholder.replace_all_uses_with(deduped_states_dict[state])
            gm.graph.erase_node(placeholder)

    return gm, tuple(deduped_states_list)

def functionalize_root_GraphModule(gm: torch.fx.GraphModule, preserve_prefix_placeholder):
    for model_name in gm._modules:
        functionalize_submod(gm, model_name)
    gm, extra_states, orig_input_num = extract_states_from_GM(gm, preserve_prefix_placeholder)
    gm, extra_states = dedupe_states(gm, extra_states, preserve_prefix_placeholder, orig_input_num - preserve_prefix_placeholder)
    return gm, extra_states
