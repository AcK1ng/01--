import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer,  LlamaConfig
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset

model_path="llama2/model/Llama2-Chinese-7b-Chat/"
tokenizer = LlamaTokenizer.from_pretrained(model_path)

train_file = "llama2/data/wikitext-103-raw/wiki.test.raw"
eval_file = "llama2/data/wikitext-103-raw/wiki.valid.raw"
max_seq_length = 512
batch_size = 4

tokenizer.pad_token = tokenizer.eos_token

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=max_seq_length,
)

import torch

device = "cuda"

class ModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = LlamaConfig.from_pretrained("llama2/model/Llama2-Chinese-7b-Chat/config.json")
        self.model = LlamaForCausalLM(self.config)
    def forward(self, input_ids, attn_mask, labels):
        ret = self.model(input_ids = input_ids,
                     attention_mask = attn_mask,
                     labels = labels,
                     use_cache = False,
                     output_hidden_states = False,
                     output_attentions = False,
                     return_dict = False)
        return ret[0]

my_model = ModelWrapper()
my_model.to(device)
print(my_model)
    
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._dynamo.source import ConstantSource
shape_env = ShapeEnv()

batch_size_source = ConstantSource(f"batchsize")
batch_size = shape_env.create_symintnode(
    shape_env.create_symbol(4, batch_size_source),
    hint=4,
    source=batch_size_source
)

max_len_source = ConstantSource(f"maxlen")
max_len = shape_env.create_symintnode(
    shape_env.create_symbol(248, max_len_source),
    hint=248,
    source=max_len_source
)

with FakeTensorMode(static_shapes = False, shape_env = shape_env) as fake_mode:
    _input_ids = torch.empty((batch_size, max_len), dtype = torch.int64).to(device)
    _attn_mask = torch.empty((batch_size, max_len), dtype = torch.int64).to(device)
    _lables = torch.empty((batch_size, max_len), dtype = torch.int64).to(device)
    args = (_input_ids, _attn_mask, _lables)
    import my_graph_capture
    if device == "cpu":
        intermediate = my_graph_capture.capture_training_graph_stage_1(my_model,
                                                             torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
                                                             8,
                                                             "SGD",
                                                             args = args)
    else:
        intermediate = my_graph_capture.capture_training_graph_stage_1(my_model,
                                                             torch.ops.aten._scaled_dot_product_efficient_attention.default,
                                                             8,
                                                             "SGD",
                                                             args = args)

graph_pack = my_graph_capture.capture_training_graph_stage_2(intermediate)
        
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 4, collate_fn = data_collator)
data = next(iter(dataloader))
example_inputs = (data["input_ids"].to(device), data["attention_mask"].to(device), data["labels"].to(device))

graph_pack["example_inputs"] = example_inputs

my_graph_capture.save_to(graph_pack, "llama2_grads_TS")
example_inputs = graph_pack["example_inputs"]
fw_output = my_graph_capture.native_debug_run(graph_pack, device, bw = False)
# bw_output = my_graph_capture.native_debug_run(graph_pack, device, bw = True)
output = my_model(*example_inputs)
output.backward()
pass

