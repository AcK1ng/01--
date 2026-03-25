#!/bin/env python
from __future__ import annotations

import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
    Any,
    Callable,
    cast,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
import pickle

import torch
import torch.jit._trace
import torch.serialization
from torch import _C

def _is_constant_tensor_list(node):
    if node.kind() != "prim::Constant":
        return False
    output_type = node.output().type()
    if output_type.isSubtypeOf(_C.ListType.ofTensors()):
        return True
    if output_type.isSubtypeOf(_C.ListType(_C.OptionalType.ofTensor())):
        return True

def _split_tensor_list_constants(g, block):
    for node in block.nodes():
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        if _is_constant_tensor_list(node):
            inputs = []
            for val in node.output().toIValue():
                input = g.insertConstant(val)
                input.node().moveBefore(node)
                input.node().copyMetadata(node)
                inputs.append(input)

            lc = (
                g.create("prim::ListConstruct", inputs)
                .insertBefore(node)
                .output()
                .setType(_C.ListType.ofTensors())
            )
            lc.node().copyMetadata(node)
            node.output().replaceAllUsesWith(lc)

# 自己实现一个常量传播
# 和官方_jit_pass_constant_propagation的区别在于
# 我不想执行算子库的算子
# 否则会生成Tensor prim::Constant，不利于序列化
# 而且要避免 %18 : bool[] = prim::Constant[value=[1, 1, 1]]()，会有运行上的bug
def constant_propagation(graph: torch.Graph):
    nodes = list(graph.nodes())
    graph.setInsertPoint(nodes[0])

    serialized_const__to__value = {}
    value__to__const = {}
    for node in nodes:
        is_input_const = [input in value__to__const for input in node.inputs()]
        all_input_consts = all(is_input_const)
        if not all_input_consts:
            continue
        actual_inputs = [value__to__const[input] for input in node.inputs()]
        if node.kind() == "prim::Constant" and node.hasAttribute("value"):
            const = node.output().toIValue()
            if isinstance(node.output().type(), torch.BoolType):
                const = True if const else False
        elif node.kind() == "prim::Constant" and not node.hasAttribute("value"):
            const = None
        elif node.kind() == "prim::ListUnpack":
            const = actual_inputs[0][actual_inputs[1]]
        elif node.kind() == "prim::TupleConstruct":
            is_bool_elem = [isinstance(elem, torch.BoolType) for elem in node.output().type().elements()]
            has_bool_elem = any(is_bool_elem)
            if has_bool_elem:
                continue
            const = tuple(actual_inputs)
        elif node.kind() == "prim::ListConstruct":
            if isinstance(node.output().type().getElementType(), torch.BoolType):
                continue
            const = actual_inputs
        else:
            continue
        serialized_const = pickle.dumps(const)
        if serialized_const not in serialized_const__to__value:
            regular_value = graph.insertConstant(const)
            regular_value.setTypeAs(node.output())
            serialized_const__to__value[serialized_const] = regular_value
            value__to__const[regular_value] = const
        node.output().replaceAllUsesWith(serialized_const__to__value[serialized_const])
        pass
    pass

def dce(graph: torch.Graph):
    nodes = list(graph.nodes())

    for node in reversed(nodes):
        if node.hasUses():
            continue

        if node.kind() == "aten::copy_":
            continue
        node.destroy()
        pass


def _optimize_graph(graph: _C.Graph):


    import os
    has_type_verbosity = "PYTORCH_JIT_TYPE_VERBOSITY" in os.environ
    
    if has_type_verbosity:
        orig_type_verbosity = os.environ["PYTORCH_JIT_TYPE_VERBOSITY"]
        os.environ["PYTORCH_JIT_TYPE_VERBOSITY"] = "1"


    # 而且fake tensor mode下时，有一些SymInt类型，会引发找不到对应的算子错误；
    # 但在torch script这里我们已经不需要了，所以走一遍序列化把SymInt转回Int
    graph = torch.parse_ir(graph.__repr__())

    # _C._jit_pass_inline(graph)
    # _C._jit_pass_inline_fork_wait(graph)
    # _C._jit_pass_constant_propagation(graph)
    constant_propagation(graph)
    _split_tensor_list_constants(graph, graph)
    dce(graph)
    # _C._jit_pass_dce(graph)
    # _C._jit_pass_canonicalize_graph_fuser_ops(graph)

    # 走一遍序列化，不这样的话窥孔优化pass会段错误，不知道怎么回事
    graph = torch.parse_ir(graph.__repr__())
    # _C._jit_pass_peephole(graph, True)

    # _C._jit_pass_fuse_addmm(graph)
    # _C._jit_pass_peephole(graph, True)
    # graph = _C._jit_pass_canonicalize(graph)
    _C._jit_pass_lint(graph)
    
    if has_type_verbosity:
        os.environ["PYTORCH_JIT_TYPE_VERBOSITY"] = orig_type_verbosity

    return graph



if __name__ == "__main__":
    with open("test.log", "r") as f:
        ts_str = f.read()
    ts = torch.parse_ir(ts_str)
    _optimize_graph(ts)
    pass
    import graph_export
    graph_pack = graph_export.read_from_no_states("llama2_grads_TS")
    ts_mod = torch.parse_ir(graph_pack["ts_fw_root"])
    ts_submods = {}
    for submod_name, submod_str in graph_pack["ts_fw_submods"].items():
        ts_submods[submod_name] = torch.parse_ir(submod_str)
    
    
    for submod in ts_submods.values():
        constant_propagation(submod)
    
    pass