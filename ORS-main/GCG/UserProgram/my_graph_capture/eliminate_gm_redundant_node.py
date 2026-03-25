import torch
import operator
from typing import Tuple, List, Dict


def eliminate_redundant_getitem(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in reversed(gm.graph.nodes):
        if node.op == "call_function" and node.target == operator.getitem:
            if isinstance(node.args[0], tuple):
                node.replace_all_uses_with(node.args[0][node.args[1]])
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def eliminate_redundant_detach(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in reversed(gm.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.detach.default:
            node.replace_all_uses_with(node.args[0])
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def eliminate_assert(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in reversed(gm.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten._assert_tensor_metadata.default:
            gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm