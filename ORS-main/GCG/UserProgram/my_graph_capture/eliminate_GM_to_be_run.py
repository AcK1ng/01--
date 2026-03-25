import torch

def change_view_to_reshape(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.view.default:
            node.target = torch.ops.aten.reshape.default
    gm.recompile()
    return gm

def add_device_kwargs__for__some_node(gm, device):
    arange_nodes = [node for node in gm.graph.nodes if node.op == "call_function" and
                                                       (node.target == torch.ops.aten.arange.start or
                                                       node.target == torch.ops.aten.arange.default or
                                                       node.target == torch.ops.aten.full.default or
                                                       node.target == torch.ops.aten.zeros.default)]
    for node in arange_nodes:
        old_kw = dict(node.kwargs)
        old_kw["device"] = torch.device(device)
        node.kwargs = old_kw
    return gm

def eliminate__device_in_kwargs(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and "device" in node.kwargs:
            new_kwargs = dict()
            for k, v in node.kwargs.items():
                if k != "device":
                    new_kwargs[k] = v
            node.kwargs = new_kwargs
    return gm

def eliminate_to_device(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.op == "call_method":
            if node.target == "to":
                if "device" in node.kwargs:
                    kwargs = node.kwargs
                    del kwargs["device"]
                    node.kwargs = kwargs
                # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
                # Tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
                if 2 == len(node.args) and (isinstance(node.args[1], str) or isinstance(node.args[1], torch.device)):
                    node.args = (node.args[0],)
                elif 2 < len(node.args) and (isinstance(node.args[1], str) or isinstance(node.args[1], torch.device)):
                    args = list(node.args)
                    args[1] = None
                    node.args = tuple(args)
                if 0 == len(node.kwargs) and 1 == len(node.args):
                    node.replace_all_uses_with(node.args[0])
                    gm.graph.erase_node(node)


    return gm