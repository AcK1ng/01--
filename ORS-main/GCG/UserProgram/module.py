#!/bin/python

import torch.nn as nn
import torch
import my_graph_capture

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # from https://arxiv.org/abs/1202.2745
        self.module = nn.Sequential(nn.Conv2d(1, 20, 4),
				    nn.MaxPool2d(2),
				    nn.Conv2d(20, 40, 5),
				    nn.MaxPool2d(3),
				    nn.Flatten(),
				    nn.Linear(40 * 3 * 3, 150),
				    nn.Linear(150, 10))
        
        self.loss = nn.MSELoss()

    def forward(self, x, target):
        pred = self.module(x)
        loss = self.loss(pred, target)
        return loss

if __name__ == '__main__':
    print("Test the module")
    device = "cuda"
    my_module = MyModule()
    my_module.train()
    my_module.to(device)

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch._dynamo.source import ConstantSource

    shape_env = ShapeEnv()

    example_bsz = 4
    source = ConstantSource(f"bsz")
    bsz = shape_env.create_symintnode(
        shape_env.create_symbol(example_bsz, source),
        hint=example_bsz,
        source=source
    )

    with FakeTensorMode(static_shapes = False, shape_env = shape_env) as fake_mode:
        _input = torch.randn(bsz, 1, 29, 29).to(device)
        _target = torch.randn(bsz, 10).to(device)
        args = (_input, _target)

        intermediate = my_graph_capture.capture_training_graph_stage_1(my_module,
                                                             torch.ops.aten.max_pool2d_with_indices.default,
                                                             2,
                                                             "SGD",
                                                             args = args)

    graph_pack = my_graph_capture.capture_training_graph_stage_2(intermediate)
        
    example_input = torch.ones((example_bsz, 1, 29, 29)).to(device)
    example_target = torch.randn(example_bsz, 10).to(device)
    example_input = (example_input, example_target)

    graph_pack["example_inputs"] = example_input
    my_graph_capture.save_to(graph_pack, "test_module")

    bw_output = my_graph_capture.native_debug_run(graph_pack, device, bw = True)
    fw_output = my_graph_capture.native_debug_run(graph_pack, device, bw = False)

    loss = my_module(*example_input)
    loss.backward()
    params = list(my_module.parameters())

    print(loss)
else:
    print("Import module!")

