import my_graph_capture


graph_pack = my_graph_capture.read_from_dummy_state("llama3_infer_TS")

fw_output = my_graph_capture.native_debug_run(graph_pack, "cpu", bw = False)
# example_inputs = list(graph_pack["example_inputs"])
# example_inputs[32] = fw_output[0]
# example_inputs[33] = fw_output[1]
# graph_pack["example_inputs"] = tuple(example_inputs)
# fw_output = my_graph_capture.native_debug_run(graph_pack, "cpu", bw = False)
# pass