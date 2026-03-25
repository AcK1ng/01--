#!/bin/env python
import torch
from typing import Tuple
import operator
import torch.utils._pytree as pytree
from torch.fx.traceback import get_current_meta, preserve_node_meta


class TS_translater(torch.fx.passes.shape_prop.ShapeProp):
    def __init__(self, gm):
        super().__init__(gm)
        self.ts = torch.Graph()
        self.out__to__ts_value = {}

        self.ts_submods = {}

        self.symexpr__to__symbol = {}
        self.symbol__to__symexpr = {}

    def value_to_type(self, value):
        if isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            shape = []
            for _dim in value.shape:
                dim = _dim
                if not isinstance(_dim, int):
                    pass
                if isinstance(_dim, torch.SymInt):
                    symint = _dim.__repr__()
                    if symint not in self.symexpr__to__symbol:
                        symbol = torch._C._new_symbolic_shape_symbol()
                        self.symexpr__to__symbol[symint] = symbol
                        self.symbol__to__symexpr[symbol] = symint
                    dim = self.symexpr__to__symbol[symint]
                shape.append(dim)
            t = torch.TensorType.get().with_dtype(value.dtype).with_sizes(shape)
            return t
        elif isinstance(value, tuple):
            return torch.TupleType([self.value_to_type(v) for v in value])
        elif isinstance(value, list):
            return torch.ListType(self.value_to_type(value[0]))
        elif isinstance(value, int):
            return torch.IntType.get()
        elif isinstance(value, float):
            return torch.FloatType.get()
        elif isinstance(value, torch.SymInt):
            symint = value.__repr__()
            if symint not in self.symexpr__to__symbol:
                symbol = torch._C._new_symbolic_shape_symbol()
                self.symexpr__to__symbol[symint] = symbol
                self.symbol__to__symexpr[symbol] = symint
            return torch.SymIntType.get()
        elif value is None:
            return torch.NoneType.get()
        else:
            assert False # unsupported

    def add_Fxnode__to__ts_value(self, fx_node, ts_value):
        meta = get_current_meta()
        
        flatted, flatted_spec = pytree.tree_flatten(fx_node)
        new_flatted = []
        for elem in flatted:
            t = elem
            if isinstance(elem, torch.SymInt):
                t = elem.__repr__()
            new_flatted.append(t)
        unflatted = pytree.tree_unflatten(new_flatted, flatted_spec)
        self.out__to__ts_value[unflatted] = ts_value
        pass

    def arg_to_value(self, arg):
        
        flatted, flatted_spec = pytree.tree_flatten(arg)
        new_flatted = []
        has_sym = False
        for elem in flatted:
            t = elem
            if isinstance(elem, torch.SymInt):
                t = elem.__repr__()
                has_sym = True
            new_flatted.append(t)
        unflatted = pytree.tree_unflatten(new_flatted, flatted_spec)

        if unflatted in self.out__to__ts_value:
            return self.out__to__ts_value[unflatted]
        elif isinstance(unflatted, list):
            inputs = [self.arg_to_value(elem) for elem in arg]
            node = self.ts.create("prim::ListConstruct", inputs)
            self.ts.insertNode(node)
            
            out = node.output()
            out.setType(torch.ListType(inputs[0].type()))

            return out
        else:
            return self.ts.insertConstant(unflatted)
    

    def get_attr(self, target, args, kwargs):
        assert False
    def call_method(self, target, args, kwargs):
        assert False

    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        input_value = self.ts.addInput(target)
        input_value.setType(self.value_to_type(out))
        self.add_Fxnode__to__ts_value(out, input_value)
        return out

    def call_function(self, target, args, kwargs):
        out = super().call_function(target, args, kwargs)

        if target == torch.ops.aten.reshape.default:
            pass

        input_values = [self.arg_to_value(arg) for arg in args]
        
        if target == operator.getitem:
            assert isinstance(args[1], int)
            input_values[1] = self.ts.insertConstant(args[1])
            node = self.ts.create("prim::TupleIndex", input_values, 1)
            idx = input_values[1].node().i("value")
            out_value = node.outputsAt(0)
            out_value.setType(list(input_values[0].type().elements())[idx])
            out_value.setDebugName(target.__name__)
            self.add_Fxnode__to__ts_value(out, out_value)
            self.ts.insertNode(node)
        elif target == operator.add:
            assert isinstance(args[0], torch.SymInt)
            assert isinstance(args[1], torch.SymInt)

            # 这里要把输入的类型改成int，从而可以调用到aten::add.int
            # 否则的话，输出是tensor，和后面的类型对不起来，运行时会报错
            input_values[0].setType(torch.IntType.get())
            input_values[1].setType(torch.IntType.get())
            node = self.ts.create("aten::add", input_values, 1)
            out_value = node.outputsAt(0)
            out_value.setDebugName(target.__name__)
            self.ts.insertNode(node)
            out_value.setTypeAs(input_values[0])
            self.add_Fxnode__to__ts_value(out, out_value)
        elif isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
            if isinstance(target, torch._ops.OpOverloadPacket):
                def find_matching_schema(op_packet, *args, **kwargs):
                    from torch.fx.operator_schemas import get_signature_for_torch_op
                    signatures, schemas = get_signature_for_torch_op(op_packet, return_schemas = True)
                    
                    if not signatures:
                        return None, None
                    
                    # 尝试绑定参数找到匹配的签名
                    for sig, schema in zip(signatures, schemas):
                        try:
                            sig.bind(*args, **kwargs)
                            # 如果绑定成功，这个签名很可能就是我们要的
                            return sig, schema
                        except TypeError:
                            continue
                    
                    return None, None
                
                sig, schema = find_matching_schema(target, *args)
                
                target = getattr(target, schema.overload_name)


            target_params = target._schema.arguments
            for i in range(len(target_params)):
                if i < len(input_values):
                    continue
                target_param = target_params[i]
                if target_param.name in kwargs:
                    the_arg_value = self.arg_to_value(kwargs[target_param.name])
                else:
                    assert target_param.has_default_value()
                    the_arg_value = self.ts.insertConstant(target_param.default_value)

                input_values.append(the_arg_value)
            node = self.ts.create(target.overloadpacket._qualified_op_name, input_values, len(target._schema.returns))
            self.ts.insertNode(node)
            if 1 == len(target._schema.returns):
                output_value = list(node.outputs())[0]
                output_value.setType(self.value_to_type(out))
                # output_value.setType(target._schema.returns[0].type)
                self.add_Fxnode__to__ts_value(out, output_value)
            elif 1 < len(target._schema.returns):
                output_values = list(node.outputs())
                for output_value, return_Arg, out_state in zip(output_values, target._schema.returns, out):
                    # output_value.setType(return_Arg.type)
                    if target._schema.__repr__() == "aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)":
                        if output_value == output_values[0]:
                            output_value.setType(torch.TensorType.get())
                            continue
                    output_value.setType(self.value_to_type(out_state))
                    self.add_Fxnode__to__ts_value(out_state, output_value)
                
                merged_node = self.ts.create("prim::TupleConstruct", output_values, 1)
                self.ts.insertNode(merged_node)
                merged_out_value = merged_node.outputsAt(0)
                merged_out_value.setType(torch.TupleType([output_value.type() for output_value in output_values]))
                merged_out_value.setDebugName(target.__name__)
                self.add_Fxnode__to__ts_value(out, merged_out_value)
            else:
                assert False
        else:
            print(target)
            assert False # unsupport

        return out

    def call_module(self, target, args, kwargs):
        submod_gm = getattr(self.module, target)
        submod_gm.graph.lint()
        submod_gm.recompile()

        if False:
            from contextlib import contextmanager
            from torch._functorch.compile_utils import strip_overloads
            @contextmanager
            def _disable_jit_autocast():
                old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)
                try:
                    yield
                finally:
                    torch._C._jit_set_autocast_mode(old_jit_autocast_flag)

            with _disable_jit_autocast():
                strip_overloads(submod_gm)

                for node in submod_gm.graph.nodes:
                    if (
                        node.target == torch.ops.aten._to_copy
                        and len(node.args) == 1
                        and len(node.kwargs) == 1
                        and "dtype" in node.kwargs
                    ):
                        node.target = torch.ops.aten.to

                for node in submod_gm.graph.nodes:
                    new_kwargs = {}
                    for k, v in node.kwargs.items():
                        if isinstance(v, torch.device):
                            v = v.type
                        new_kwargs[k] = v
                    node.kwargs = new_kwargs

                # 用jit script会丢掉shape中的symbolic info
                example_inputs = args

                f = torch.jit.script(submod_gm, example_inputs = [example_inputs])

            submod_ts = f.inlined_graph
            submod_ts.eraseInput(0) # delete the first param 'self'
            out = super().call_module(target, args, kwargs)
        else:
            # 走这个分支的话，好像torch script的类型信息是错的
            translater = TS_translater(submod_gm)
            out = translater.propagate(*args, **kwargs)
            submod_ts = translater.ts
            output_values = list(submod_ts.outputs())
            if 1 < len(output_values):
                merged_node = submod_ts.create("prim::TupleConstruct", output_values, 1)
                submod_ts.insertNode(merged_node)
                merged_out_value = merged_node.outputsAt(0)
                merged_out_value.setType(torch.TupleType([output_value.type() for output_value in output_values]))
                [submod_ts.eraseOutput(0) for _ in range(len(output_values))]
                submod_ts.registerOutput(merged_out_value)
                


        input_values = [self.arg_to_value(arg) for arg in args]
        for arg, param in zip(input_values, list(submod_ts.inputs())):
            param.setTypeAs(arg)

        node = self.ts.create("prim::GCG_Call_Submod", input_values, 1)
        node.s_("target", target)
        self.ts.insertNode(node)

        the_value = node.outputsAt(0)
        the_value.setType(self.value_to_type(out))
        next(iter(submod_ts.outputs())).setTypeAs(the_value)
        output_node__in__submod = list(submod_ts.outputs())
        assert 1 == len(output_node__in__submod)
        output_node__in__submod = output_node__in__submod[0].node()
        if output_node__in__submod.kind() == "prim::TupleConstruct":
            for input_value, input_type in zip(output_node__in__submod.inputs(),
                                               output_node__in__submod.output().type().elements()):
                input_value.setType(input_type)
        the_value.setDebugName(target)
        self.add_Fxnode__to__ts_value(out, the_value)

        self.ts_submods[target] = submod_ts
        return out

    def output(self, target, args, kwargs):
        if isinstance(args[0], (list, tuple)):
            input_values = [self.arg_to_value(arg) for arg in args[0]]
            for value in input_values:
                self.ts.registerOutput(value)
        class Dummy:
            def __init__(self):
                pass
        self.graph._codegen.pytree_info = Dummy()
        _, self.graph._codegen.pytree_info.out_spec = pytree.tree_flatten(args)
        return super().output(target, args, kwargs)



if __name__ == "__main__":
    import pickle

    with open('tmp_submods.pickle', 'rb') as f:
        submod_graphs_str = pickle.load(f)
        submod = torch.parse_ir(submod_graphs_str)

def GraphModule_to_TorchScript(gm: torch.fx.GraphModule, *example_inputs):

    # 比如torch.aten.ops.arange用的torch.set_default_device，其输出不会根据输入的device改变
    torch.set_default_device(example_inputs[0].device)
    translater = TS_translater(gm)
    with preserve_node_meta():
        translater.propagate(*example_inputs)

    return translater.ts, translater.ts_submods, translater.symbol__to__symexpr