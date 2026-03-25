#!/bin/env python


from typing import List, Optional
from llama3_8b_instruct_infer import Dialog, Llama
import os
import torch

ckpt_dir: str = "llama3_8b_instruct_infer/"
tokenizer_path: str = "llama3_8b_instruct_infer/tokenizer.model"
temperature: float = 0.6
top_p: float = 0.9
max_seq_len: int = 16*1024

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    temperature=temperature,
    top_p=top_p
)

import time
s = time.time()
kvcaches = generator.generate_kvcaches(128)
for _ in range(1000):
    token = generator.my_generate(kvcaches, 0, [0] * 1, 1)
e = time.time()
print("Time: ", e-s)

exit()