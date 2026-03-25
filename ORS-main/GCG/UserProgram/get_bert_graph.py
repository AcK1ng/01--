#!/bin/env python
import torch
import bert.train_bert

mask_prob: float = 0.15
random_replace_prob: float = 0.1
unmask_replace_prob: float = 0.1
max_seq_length: int = 512
tokenizer: str = "roberta-base"

num_layers: int = 6
num_heads: int = 8
ff_dim: int = 512
h_dim: int = 256
dropout: float = 0.1

batch_size: int = 8

data_iterator = bert.train_bert.create_data_iterator(
    mask_prob=mask_prob,
    random_replace_prob=random_replace_prob,
    unmask_replace_prob=unmask_replace_prob,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    batch_size=batch_size,
    directory = "bert"
)

model = bert.train_bert.create_model(
    num_layers=num_layers,
    num_heads=num_heads,
    ff_dim=ff_dim,
    h_dim=h_dim,
    dropout=dropout,
)

batch = next(iter(data_iterator))

example_inputs = (batch["src_tokens"], batch["attention_mask"], batch["tgt_tokens"])


output = model(*example_inputs)

from alpa_for_pytorch import my_graph_capture
orig_module_str, ts_mod, ts_submods, states = my_graph_capture.capture_backward(model, torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default, *example_inputs)
my_graph_capture.save_to(orig_module_str, ts_mod, ts_submods, states, example_inputs, "bert_grads_TS")
ts_mod, ts_submods, example_inputs, tensors_descriptors, states = my_graph_capture.read_from("bert_grads_TS")
res = my_graph_capture.native_debug_run(ts_mod, ts_submods, "cpu", *(states + example_inputs))