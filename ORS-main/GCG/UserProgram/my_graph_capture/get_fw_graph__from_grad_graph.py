import torch

def get_fw_graph__from_grad_graph(root, submods, fw_output_nr):
    fw_graph = root.copy()
    fw_submods = dict()
    for name, submod in submods.items():
        fw_submods[name] = submod.copy()
    all_output_nr = len(list(fw_graph.outputs()))
    [fw_graph.eraseOutput(i) for i in reversed(range(all_output_nr - fw_output_nr))]
    return fw_graph, fw_submods