#!/bin/env python

import torch


torch.set_default_device(torch.device("cuda:0"))
t1 = torch.ops.aten.empty([3, 4])

print(t1)
print(t1.device)

t = torch.ops.aten.empty([], dtype = torch.float64, layout = torch.strided)

torch.library.define("native_worker::get_device", "() -> Device")

torch.library.impl("native_worker::get_device", "default", lambda: torch.device("cuda"))
#def get_device():
#    return torch.device("cuda")

print(torch.ops.native_worker.get_device())

empty_tensor = """
graph(%8: int):
  %7 : None = prim::Constant()
  %0 : int[] = prim::Constant[value=None]()
  %1 : int = prim::Constant[value=6]()
  %2 : int = prim::Constant[value=0]()
  %3 : NoneType = prim::Constant()
  %5 : Tensor = aten::empty(%0, %1, %2, %7, %3, %3)
  return (%5)
"""

res = torch._C._jit_interpret_graph(torch.parse_ir(empty_tensor), (1, ))
#res = torch._C._jit_interpret_graph(torch.parse_ir(empty_tensor, parse_tensor_constants = True), ([3, 4], torch.float64, torch.strided))
#res = torch._C._jit_interpret_graph(torch.parse_ir(zero_tensor, parse_tensor_constants = True), ())

print(res.shape.__str__()[11:-1])

def scalar_type_to_int(scalar_type):
    return torch._C._jit_interpret_graph(torch.parse_ir("""
graph(%8: dtype):
  return (%8)
"""), (scalar_type, ))

print(scalar_type_to_int(torch.float))
print(scalar_type_to_int(torch.strided))

