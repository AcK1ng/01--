import torch

def eliminate_wrong_type(ts: torch.Graph) -> torch.Graph:
    for node in ts.nodes():
        for value in node.outputs():
            type = value.type()
            if "NamedTuple" in type.str():
                def is_type_NamedTuple(type):
                    return type.str().startswith("NamedTuple")

                def fix_nest_NamedTuple(parent_type):
                    if parent_type.kind() == "TupleType":
                        if is_type_NamedTuple(type):
                            return torch.TupleType(type.elements())
                        else:
                            elements = [fix_nest_NamedTuple(e) for e in parent_type.elements()]
                            return torch.TupleType(elements)
                    else:
                        return parent_type

                value.setType(fix_nest_NamedTuple(type))
                
    return ts

def eliminate_literalTrue(ts: torch.Graph) -> torch.Graph:
    const_1_value = None
    def get_const_1():
        nonlocal const_1_value
        if const_1_value == None:
            for node in ts.nodes():
                if node.kind() == "prim::Constant" and \
                   node.outputsAt(0).type().kind() == "BoolType" and \
                   node.hasAttribute("value") and \
                   node.kindOf("value") == "i" and \
                   node.i("value") == 1:
                    const_1_value = node.outputsAt(0)
                    break
            if const_1_value == None:
                the_node = ts.create("prim::Constant", [], 1)
                the_node.i_("value", 1)
                ts.prependNode(the_node)

                const_1_value = the_node.outputsAt(0)
                const_1_value.setType(torch.BoolType.get())
        return const_1_value


    const_0_value = None
    def get_const_0():
        nonlocal const_0_value
        if const_0_value == None:
            for node in ts.nodes():
                if node.kind() == "prim::Constant" and \
                   node.outputsAt(0).type().kind() == "BoolType" and \
                   node.hasAttribute("value") and \
                   node.kindOf("value") == "i" and \
                   node.i("value") == 0:
                    const_0_value = node.outputsAt(0)
                    break
            if const_0_value == None:
                the_node = ts.create("prim::Constant", [], 1)
                the_node.i_("value", 0)
                ts.prependNode(the_node)

                const_0_value = the_node.outputsAt(0)
                const_0_value.setType(torch.BoolType.get())
        return const_0_value

    for node in ts.nodes():
        if node.kind() != "prim::Constant":
            continue
        if not node.hasAttribute("value"):
            continue
        if node.kindOf("value") != "ival":
            continue
        the_constant = node.ival("value")

        has_literalTrue = False
        orig_flatted, tree_spec = torch.utils._pytree.tree_flatten(the_constant)
        for i in range(len(orig_flatted)):
            orig_e = orig_flatted[i]
            if isinstance(orig_e, bool):
                has_literalTrue = True
                break
        if not has_literalTrue:
            continue

        orig_value = node.outputsAt(0)

        if orig_value.type().kind() == "ListType" and \
            orig_value.type().containedTypes()[0].kind() == "BoolType":
            replaced_node = ts.create("prim::ListConstruct", [get_const_1() if b == True
                                                              else get_const_0() for b in the_constant], 1)
            ts.setInsertPoint(node)
            ts.insertNode(replaced_node)

            replaced_value = replaced_node.outputsAt(0)
            replaced_value.setTypeAs(orig_value)
            orig_value.replaceAllUsesWith(replaced_value)
        else:
            assert False # not implemented

    torch._C._jit_pass_dce(ts)
            
    return ts




# 因为有些算子一定要device，所以在GraphModule的时候为了抓计算图我们加进去
# 但是在这里我们替换成适合在云端运行的get_device
def replace_device(ts: torch.Graph):
    device_nodes = []
    for node in ts.nodes():
        if node.outputsSize() != 1:
            # ignore
            continue

        if node.output().type().kind() == "DeviceObjType":
            device_nodes.append(node)

    if 0 == len(device_nodes):
        return ts
    
    dev_node = ts.create("prim::GCG_get_native_device", [], 1)
    ts.setInsertPoint(device_nodes[0])
    ts.insertNode(dev_node)
    dev_value = dev_node.output()
    dev_value.setTypeAs(device_nodes[0].output())

    for node in device_nodes:
        node.output().replaceAllUsesWith(dev_value)
    return ts

def fix_type_for_some_op__to_run(ts: torch.Graph) -> torch.Graph:
    for node in ts.nodes():
        if node.kind() == "aten::hstack":
            input_value = node.inputsAt(0)
            input_value.setType(torch.ListType(torch.TensorType.get())) # hstack的type如果是Float(SS(-11), SS(-12))[]会报错
            
        if node.kind() == "aten::index_put_":
            input_value = node.inputsAt(1)
            input_value.setType(torch.ListType(torch.TensorType.get())) # 如果是Bool(1, SS(-213))[]会报错
    return ts

if __name__ == "__main__":
    with open("text1.log", "r") as f:
        ts_str = f.read()
    ts = torch.parse_ir(ts_str)
    ts = replace_device(ts)
    pass