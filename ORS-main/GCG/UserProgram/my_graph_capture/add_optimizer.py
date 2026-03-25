import torch



def add_SGD(fw_output_nr, fw_bw_gm, lr):
    fw_bw_gm_grads = fw_bw_gm.graph.output_node().args[0][:-fw_output_nr]

    fw_param_placeholders = [node for node in fw_bw_gm.graph.nodes if node.op == "placeholder"]
    fw_param_placeholders = fw_param_placeholders[:len(fw_bw_gm_grads)]

    params_after_SGD = []

    for i, (param, grad) in enumerate(zip(fw_param_placeholders, fw_bw_gm_grads)):
        with fw_bw_gm.graph.inserting_after(grad):
            t = fw_bw_gm.graph.call_function(torch.ops.aten.mul, (grad, -lr))
        with fw_bw_gm.graph.inserting_after(t):
            new_param = fw_bw_gm.graph.call_function(torch.ops.aten.add, (param, t))
            params_after_SGD.append(new_param)

    output_nodes = params_after_SGD + fw_bw_gm.graph.output_node().args[0][len(params_after_SGD):]
    fw_bw_gm.graph.erase_node(fw_bw_gm.graph.output_node())
    fw_bw_gm.graph.output(output_nodes)
    fw_bw_gm.recompile()

    return fw_bw_gm