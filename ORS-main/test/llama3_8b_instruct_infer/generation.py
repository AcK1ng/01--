# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from llama3_8b_instruct_infer.model import ModelArgs, Transformer, TransformerWrapper
from llama3_8b_instruct_infer.tokenizer import ChatFormat, Dialog, Message, Tokenizer



class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        torch.set_default_device("cuda")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        # assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        # ckpt_path = checkpoints[0]
        # checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=1,
            **params,
        ) 
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        model = TransformerWrapper(temperature, top_p, model_args)
        # model.model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer, model_args)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, params):
        self.params = params
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    def _generate_kvcaches(self, max_seq_len, n_kv_heads, head_dim, n_layers):
        return tuple([torch.empty(2, max_seq_len, n_kv_heads, head_dim) for _ in range(n_layers)])
    def generate_kvcaches(self, max_seq_len: Optional[int] = None):
        params = self.params
        if max_seq_len is None:
            max_seq_len = params.max_seq_len
        return self._generate_kvcaches(max_seq_len, params.n_kv_heads, params.dim // params.n_heads, params.n_layers)


    def get_computational_graph(self, shape_env, ):
        
        params = self.params

        from torch._dynamo.source import ConstantSource
    
        def get_int_sym(example_val: int, name: str) -> torch.SymInt:
            source = ConstantSource(name)
            return shape_env.create_symintnode(
                shape_env.create_symbol(example_val, source),
                hint=example_val,
                source=source
            )

        # max_seq_len = get_int_sym(8192, "max_seq_len")
        max_seq_len = params.max_seq_len
        n_kv_heads = params.n_kv_heads
        head_dim = params.dim // params.n_heads
        n_layers = params.n_layers

        local_hit = get_int_sym(3, "local_hit")
        miss_len = get_int_sym(12, "miss_len")

        kvcaches = self._generate_kvcaches(max_seq_len, n_kv_heads, head_dim, n_layers)
        tokens = torch.empty((1, miss_len,), dtype=torch.long)
        tensor__for_local_hit = torch.empty((local_hit,))
        ep = torch.export.export(self.model, args = (kvcaches + (tokens, tensor__for_local_hit)), strict = True)

        return ep


    def my_generate(
        self,
        kvcaches: Tuple[torch.Tensor],
        local_hit: int,
        prompt_tokens: List[int],
        max_gen_len: int = 300
    ):
        params = self.params

        pad_id = self.tokenizer.pad_id
        cur_pos__end = min(params.max_seq_len, max_gen_len + len(prompt_tokens))
        tokens = torch.full((1, cur_pos__end,), pad_id, dtype=torch.long)
        tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long)

        stop_tokens = set(self.tokenizer.stop_tokens)

        prev_pos = local_hit
        tensor_for_prev_pos = torch.empty((prev_pos,))
        next_token = tokens[:, prev_pos:len(prompt_tokens)]

        for cur_pos in range(len(prompt_tokens), cur_pos__end):
            next_token, tensor_for_prev_pos = self.model.forward(*(kvcaches + (next_token, tensor_for_prev_pos)))
            tokens[0, cur_pos] = next_token[0]
            next_token_id = next_token.item()
            if True and (next_token_id != 128009):
                print(self.tokenizer.decode([next_token_id]), end = "", flush = True)
            # prev_pos = cur_pos
            if next_token_id in stop_tokens:
                break

        return tokens.tolist()[0][:cur_pos + 1]
        pass

