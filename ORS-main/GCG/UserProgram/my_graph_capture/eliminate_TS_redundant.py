#!/bin/env python
import torch
from typing import Tuple, Dict
from collections import Counter

from . import optimize_TS
def remove_unused_submods(root, submods):
    ret_submods = dict()
    for node in root.nodes():
        if node.kind() != "prim::GCG_Call_Submod":
            continue
        submod_name = node.s("target")
        ret_submods[submod_name] = submods[submod_name]
    return root, ret_submods


# 如果submod的输出是Tuple，而且某个输出分量没人用，那么就在submod里的分量输出None
# 这要求submod的输出仅仅由TupleIndex使用，不能整个输出传给别人了。
# %29 : ((Float(*, 32, *, 128), Float(*, 32, *), Long(), Long()), Float(*, *, 4096), int) = prim::GCG_Call_Submod[target="submod_fw6"](%merged_state.13, %18, %27)
# 比如最后的那个int
def set_unsed_submod_output__is__None(root, submods):
    for node in root.nodes():
        if node.kind() != "prim::GCG_Call_Submod":
            continue
        assert 1 == node.outputsSize()
        root_out_value = node.outputsAt(0)
        user_kinds = set([use.user.kind() for use in root_out_value.uses()])
        user_kinds.discard("prim::TupleIndex")
        if 0 < len(user_kinds):
            # has user kind other than prim::TupleIndex
            continue

        used_idxes = set([use.user.inputsAt(1).node().i("value") for use in root_out_value.uses()])
        all_idxes = set(range(len(root_out_value.type().elements())))
        unused_idxes = all_idxes - used_idxes

        if 0 == len(unused_idxes):
            continue

        submod = submods[node.s("target")]
        none_node = submod.create("prim::Constant", [], 1)
        submod.prependNode(none_node)

        none_value = none_node.outputsAt(0)
        none_value.setType(torch.NoneType.get())

        submod_out_value = list(submod.outputs())[0]
        submod_out_node = submod_out_value.node()
        [submod_out_node.replaceInput(unused_idx, none_value) for unused_idx in unused_idxes]
        submod_out_value.setType(torch.TupleType([value.type() for value in submod_out_node.inputs()]))
        root_out_value.setTypeAs(submod_out_value)

        pass
    return root, submods


def merge_constant__for_my_root(root: torch.Graph):
    orig_constant_nodes = [node for node in root.nodes() if node.kind() == "prim::Constant" and node.hasAttribute("value") and node.kindOf("value") == "i"]
    constant_to_values = dict()
    for node in orig_constant_nodes:
        constant = node.i("value")
        if constant not in constant_to_values:
            constant_to_values[constant] = set()
        constant_to_values[constant].add(node.outputsAt(0))

    merge_chances = dict()
    for constant, values in constant_to_values.items():
        if 1 == len(values):
            continue
        merge_chances[constant] = values

    dummy_node = root.create("prim::Constant", [], 1)
    root.prependNode(dummy_node)
    root.setInsertPoint(dummy_node)
    for constant, orig_values in merge_chances.items():
        new_value = root.insertConstant(constant)

        for orig_value in orig_values:
            orig_value.replaceAllUsesWith(new_value)

    return root



def eliminate_useless_placeholder_of_submods(ts_module: torch.Graph, ts_submods: Dict[str, torch.Graph]):
    for submod_name in ts_submods:
        submod = ts_submods[submod_name]
        optimize_TS.dce(submod)

        formal_arg_values = submod.inputs()
        args_useless = [False if 0 < len(value.uses())
                               else True for value in formal_arg_values]
        useless_args_index = [arg_index
                              for arg_index, arg_useless in zip(reversed(range(len(args_useless))), reversed(args_useless))
                              if arg_useless]


        if len(useless_args_index) == 0:
            continue

        #print("________________________________")
        #print(submod)
        for arg_index in useless_args_index:
            submod.eraseInput(arg_index)
        #print("________________________________")
        #print(submod)

        call_nodes = [node for node in ts_module.nodes() if node.kind() == "prim::GCG_Call_Submod" and node.s("target") == submod_name]
        for call_submod_node in call_nodes:
            for arg_index in useless_args_index:
                call_submod_node.removeInput(arg_index)

        #print("________________________________")
        #print(ts_module)

    #print("________________________________")
    #print(ts_module)
    return ts_module, ts_submods



def merge_TupleConstruct_for_root_out(root: torch.Graph, submods: Dict):
    source_values = __find__TupleIndex_values__generated_from_a_single_acc(root)
    outs = list(root.outputs())
    out_values__from_TupleConstruct = [value for value in source_values if value.node().kind() == "prim::TupleConstruct" and value in outs]
    for root_out_value in out_values__from_TupleConstruct:
        submod__root_calling_node = source_values[root_out_value]
        submod_out_value = submod__root_calling_node.outputsAt(0)

        # 我们希望submod的输出，先走一层TupleIndex，然后唯一进入这个TupleConstruct，这个时候我们才能合并
        can_be_merged = True
        for t in root_out_value.node().inputs():
            if t.node().kind() != "prim::TupleIndex":
                can_be_merged = False
                break

            if 1 < len(t.uses()):
                can_be_merged = False
                break

            if t.node().inputsAt(0) != submod_out_value:
                can_be_merged = False
                break

        if not can_be_merged:
            continue

        out_idxes = [v.node().inputsAt(1).node().i("value") for v in root_out_value.node().inputs()]



        # modify the source submod
        submod = submods[submod__root_calling_node.s("target")]
        out_node__in_submod = list(submod.outputs())[0].node()
        submod.setInsertPoint(out_node__in_submod)

        new_merged_node = submod.create("prim::TupleConstruct", [out_node__in_submod.inputsAt(out_idx) for out_idx in out_idxes], 1)
        submod.insertNode(new_merged_node)
        new_merged_value = new_merged_node.outputsAt(0)
        new_merged_value.setType(torch.TupleType([out_node__in_submod.inputsAt(out_idx).type() for out_idx in out_idxes]))

        out_node__in_submod.addInput(new_merged_value)
        out_node__in_submod.outputsAt(0).setType(torch.TupleType([v.type() for v in out_node__in_submod.inputs()]))



        # modify the root
        submod__root_calling_node.outputsAt(0).setTypeAs(out_node__in_submod.outputsAt(0))
        root.setInsertPoint(root.return_node())
        idx_value = root.insertConstant(out_node__in_submod.inputsSize() - 1)
        new_merged_node_in_root = root.create("prim::TupleIndex", [submod__root_calling_node.outputsAt(0), idx_value], 1)
        root.insertNode(new_merged_node_in_root)

        new_merged_value_in_root = new_merged_node_in_root.outputsAt(0)
        new_merged_value_in_root.setTypeAs(new_merged_value)

        root_out_value.replaceAllUsesWith(new_merged_value_in_root)

        pass
    
    return root, submods

# 寻找TupleIndex和Placeholder的value，这些value只输入到submod里，不会作为输出。
# value本身已经不是tuple了，是从某个tuple中拆出来的分量；或者本来就是从输入中来的Tensor。
# 藉此分析出，哪些tuple的TupleIndex可以合进submod的输入中；或者合并root计算图的输入
def __find__TupleIndex_and_Placeholder_values__just_used_for_submod(ts: torch.Graph):
    values__just_used_for_submod = dict()
    for node in list(reversed(list(ts.nodes()))) + [ts.param_node()]:
        if node.kind() != "prim::TupleIndex" and node.kind() != "prim::Param":
            continue

        for value in node.outputs():
            users = [use.user for use in value.uses()]
            users_kind_other_than_submod = [n for n in users if n.kind() != "prim::GCG_Call_Submod"]
            if 0 == len(users) or 0 < len(users_kind_other_than_submod):
                continue

            used_submods = list(set(users)) # 去重
            used_submods.sort(key = lambda node: node.s("target")) # 归一
            values__just_used_for_submod[value] = tuple(used_submods)

    return values__just_used_for_submod


# 找到那些，不会由多个加速器生成的value；把这些value归为一类就好。
# 我们的目的是找TupleIndex合进JiQunServe_submod的机会，所以只考察穿越TupleIndex的value。
# 归为一类的value，或许可以把沿途的TupleIndex，顺着submod的输出，塞进submod，来减少计算图value的个数；
#
# 所以，初始情况——
# 一个JuQinServe_call_submod显然在一个加速器上；它的输出value肯定归为这个submod上。
def __find__TupleIndex_values__generated_from_a_single_acc(ts:torch. Graph):
    values__generated_from_a_single_acc = dict()
    for node in ts.nodes():
        if node.kind() != "prim::GCG_Call_Submod":
            continue
        assert node.outputsSize() == 1
        values__generated_from_a_single_acc[node.outputsAt(0)] = node

    for node in ts.nodes():
        if node.kind() == "prim::TupleIndex":
            in_value = node.inputsAt(0)
            if in_value not in values__generated_from_a_single_acc:
                continue

            out_value = node.outputsAt(0)

            # 此时，该TupleIndex的输出会从，输入所在的加速器上生成
            values__generated_from_a_single_acc[out_value] = values__generated_from_a_single_acc[in_value]
        elif node.kind() == "prim::TupleConstruct":
            sources = [None if in_value not in values__generated_from_a_single_acc
                       else values__generated_from_a_single_acc[in_value]
                       for in_value in node.inputs()]
            sources = set(sources)
            if 1 < len(sources):
                continue
            if None == list(sources)[0]:
                continue

            out_value = node.outputsAt(0)
            values__generated_from_a_single_acc[out_value] = list(sources)[0]

    return values__generated_from_a_single_acc


# 比如下图的primals_1和primals_2可以合到一块
# graph(%primals_1 : Tensor,
#       %primals_2 : Tensor):
#   %11 : Tensor = prim::GCG_Call_Submod[target="submod_JiQunServe_forward_0"](%primals_9, %primals_1, %primals_2)
#   %58 : Tensor = prim::GCG_Call_Submod[target="backward__submod_3"](%53, %15, %22, %primals_9, %primals_1, %primals_1, %primals_2)
#
# 这要求只有submod_JiQunServe_forward_0和backward__submod_3共同使用primals_1和primals_2；primals_1和primals_2不能参与输出
def merge_root_params(root: torch.Graph, submods: Dict,
                      unpreserved_states: Tuple,
                      preserve_prefix_placeholder,
                      preserve_suffix_placeholder,
                      sync_modify_output):
    unpreserved_root_params = [value for value in root.inputs()]
    unpreserved_root_params = unpreserved_root_params[preserve_prefix_placeholder: len(unpreserved_root_params) - preserve_suffix_placeholder]
    assert len(unpreserved_root_params) == len(unpreserved_states)
    root_params__to__args = dict(zip(unpreserved_root_params, unpreserved_states))

    if sync_modify_output:
        all_root_params = [value for value in root.inputs()]
        all_root_outputs = [value for value in root.outputs()]

        # 每个param都有对应的out value
        root_params__to__root_outputs = dict(zip(all_root_params, all_root_outputs))

        # 对于超出param数量的out value，单独记下来
        if len(all_root_params) < len(all_root_outputs):
            extra_outputs = all_root_outputs[len(all_root_params): ]
        else:
            extra_outputs = []

    def find_merge_change():
        value__to__target_submods =__find__TupleIndex_and_Placeholder_values__just_used_for_submod(root)
        if sync_modify_output:
            value__to__source_submod = __find__TupleIndex_values__generated_from_a_single_acc(root)
            params_to_be_splitted = dict()
            for param, target_submods in value__to__target_submods.items():
                if param not in unpreserved_root_params:
                    continue
                out = root_params__to__root_outputs[param]
                if out not in value__to__source_submod:
                    continue
                out_source_submod = value__to__source_submod[out]
 
                # 如果要还要考虑output来合并input，每个input value都对应一个output value
                # 对于那些要合并的input value，target_submods要共同使用这些input value；
                # 对应要合并的output value一定是从一个submod上输出的
                if (target_submods, out_source_submod) not in params_to_be_splitted:
                    params_to_be_splitted[(target_submods, out_source_submod)] = set()
                params_to_be_splitted[(target_submods, out_source_submod)].add(param)

            merge_change = None
            for (target_submods, _), params in params_to_be_splitted.items():
                if 1 < len(params):
                    merge_change = (target_submods, tuple(params))
                    break
        else:
            target_submods__to__values = dict()
            for param, target_submods in value__to__target_submods.items():
                if param not in unpreserved_root_params:
                    continue
                if target_submods not in target_submods__to__values:
                    target_submods__to__values[target_submods] = set()
                target_submods__to__values[target_submods].add(param)

            merge_change = None
            for target_submods, params in target_submods__to__values.items():
                if 1 < len(params):
                    merge_change = (target_submods, tuple(params))
                    break
        return merge_change

    merge_change = find_merge_change()
    while merge_change is not None:
        target_submods, merged_params = merge_change


        def idx_of_param(TS, param_value):
            for i, value in enumerate(TS.inputs()):
                if value == param_value:
                    return i
            assert False

        merged_states_jit_type = torch.TupleType([value.type() for value in merged_params])


        # Add an input value, who holds merged states, to root module

        #                                                 |<- preserve_suffix_placeholder ->|
        # -----------------------------------------------------------------------------------
        # | 1. Don't change! | 2. unpreserved root_params |          3. Don't change!       |
        # -----------------------------------------------------------------------------------
        #                     ^
        #         preserve_prefix_placeholder
        the_new_root_merged_param = root.addInput("merged_state")
        the_new_root_merged_param.setType(merged_states_jit_type)

        now_root_params_num = root.param_node().outputsSize()

        # Placeholders before permuting
        #                                                 |<- preserve_suffix_placeholder ->|<-      1      ->|
        # -----------------------------------------------------------------------------------------------------
        # | 1. Don't change! | 2. unpreserved root_params |          3. Don't change!       | 4. merged_state |
        # -----------------------------------------------------------------------------------------------------
        #                     ^                                                              ^                 ^
        #         preserve_prefix_placeholder                                                       now_root_params_num
        # 1. list(range(0, preserve_prefix_placeholder))
        # (2 + 3). list(range(preserve_prefix_placeholder, now_root_params_num - 1))
        # 4. list(range(now_root_params_num - 1, now_root_params_num))
        perlist = list(range(0, preserve_prefix_placeholder)) + \
                  list(range(now_root_params_num - 1, now_root_params_num)) + \
                  list(range(preserve_prefix_placeholder, now_root_params_num - 1))
        root.permuteInputs(perlist)
        # Placeholders after permuting
        #                    |<-      1      ->|                            |<- preserve_suffix_placeholder ->|
        # -----------------------------------------------------------------------------------------------------
        # | 1. Don't change! | 4. merged_state | 2. unpreserved root_params |          3. Don't change!       |
        # -----------------------------------------------------------------------------------------------------
        #                     ^                                                                                ^
        #         preserve_prefix_placeholder                                                      now_root_params_num


        the_new_root_merged_arg = tuple([root_params__to__args[value] for value in merged_params])
        root_params__to__args[the_new_root_merged_param] = the_new_root_merged_arg
        unpreserved_root_params = [the_new_root_merged_param] + unpreserved_root_params

        # 然后在root加入合并后的output value
        if sync_modify_output:
            outputs_should_be_merged = [root_params__to__root_outputs[param] for param in merged_params]
            the_new_root_merged_output_node = root.create("prim::TupleConstruct", outputs_should_be_merged, 1)
            the_new_root_merged_output = the_new_root_merged_output_node.outputsAt(0)
            the_new_root_merged_output.setType(torch.TupleType([value.type() for value in outputs_should_be_merged]))
            root_params__to__root_outputs[the_new_root_merged_param] = the_new_root_merged_output
            root.insertNode(the_new_root_merged_output_node)

        for node in target_submods:
            submod_name = node.s("target")
            submod = submods[submod_name]

            # 先修改submod中的形参列表
            submod_args = [value for value in node.inputs()]
            submod_params = [value for value in submod.inputs()]
            # 这里还有个问题就是一个实参对应多个形参的情况
            # 比如prim::GCG_Call_Submod[target="backward__submod_3"](%primals_1, %primals_1, %primals_2)
            arg__to__params = dict()
            for arg, param in zip(submod_args, submod_params):
                if arg not in merged_params:
                    continue
                if arg not in arg__to__params:
                    arg__to__params[arg] = set()
                arg__to__params[arg].add(param)

            # 把merged_state加到submod的形参里。所有原来形参的引用都重定向到merged_state上。
            merged_states_param = submod.addInput("merged_state")
            merged_states_param.setType(merged_states_jit_type)
            for idx, arg in enumerate(merged_params):
    
                idx_node = submod.create("prim::Constant", 1)
                idx_node.i_("value", idx)
                submod.prependNode(idx_node)
                idx_value = idx_node.outputsAt(0)
                idx_value.setType(torch.IntType.get())

                new_submod_param_node = submod.create("prim::TupleIndex", [merged_states_param, idx_value], 1)
                new_submod_param_node.insertAfter(idx_value.node())
                new_submod_param = new_submod_param_node.outputsAt(0)
                new_submod_param.setType(arg.type())

                for orig_submod_param in arg__to__params[arg]:
                    orig_submod_param.replaceAllUsesWith(new_submod_param)
                    submod.eraseInput(idx_of_param(submod, orig_submod_param))

            # 然后修改root中调用submod的node
            calling_args = [value for value in node.inputs() if value not in merged_params] + [the_new_root_merged_param]
            new_call_node = root.create("prim::GCG_Call_Submod", calling_args, 1)
            new_call_node.s_("target", submod_name)
            new_call_node.insertAfter(node)

            new_call_node_value = new_call_node.outputsAt(0)
            new_call_node_value.setTypeAs(node.outputsAt(0))
            node.outputsAt(0).replaceAllUsesWith(new_call_node.outputsAt(0))
            node.destroy()

        # Delete single_use_inputs and corresponding states
        for root_param in merged_params:
            unpreserved_root_params.remove(root_param)
            del root_params__to__args[root_param]
            if sync_modify_output:
                del root_params__to__root_outputs[root_param]
            root.eraseInput(idx_of_param(root, root_param))

        # 根据新input的顺序，重新register其对应的output
        if sync_modify_output:
            all_root_params = [value for value in root.inputs()]
            root.return_node().removeAllInputs()
            for param in all_root_params:

                if param not in root_params__to__root_outputs:
                    # 如果params的数量大于输出的out value数量，此时就可以停下了
                    break

                root.registerOutput(root_params__to__root_outputs[param])
            for extra_out in extra_outputs:
                root.registerOutput(extra_out)

        merge_change = find_merge_change()


    return root, submods, tuple([root_params__to__args[param] for param in unpreserved_root_params])



#  合并算子间的prim::TupleIndex
#
#  例子：
#
#  %503 : Tensor = prim::GCG_Call_Submod[target="submod_18"](%getitem_36, %getitem_1, %getitem_2, %getitem_37, %submod_18_merged_state)
#  %getitem_38 : Tensor = prim::TupleIndex(%503, %359)
#  %getitem_39 : Tensor = prim::TupleIndex(%503, %361)
#  %505 : Tensor = prim::GCG_Call_Submod[target="submod_19"](%getitem_38, %getitem_1, %getitem_2, %getitem_39, %submod_19_merged_state)
#  %506 : Tensor = prim::GCG_Call_Submod[target="submod_20"](%getitem_38, %getitem_1, %getitem_2, %getitem_39, %submod_20_merged_state)
#
#  变为
#
#  %503 : Tensor = prim::GCG_Call_Submod[target="submod_18"](%getitem_36, %getitem_1, %getitem_2, %getitem_37, %submod_18_merged_state)
#  %505 : Tensor = prim::GCG_Call_Submod[target="submod_19"](%503, %getitem_1, %getitem_2, %submod_19_merged_state)
#  %506 : Tensor = prim::GCG_Call_Submod[target="submod_20"](%503, %getitem_1, %getitem_2, %submod_20_merged_state)
#
#  若多个submod共同使用一组、由一个submod生成的prim::TupleIndex value，那就可以合并
def reduce_tupleIndex(root: torch.Graph, submods: Dict):
    def find_reduce_chance():
        value__to__target_submods =__find__TupleIndex_and_Placeholder_values__just_used_for_submod(root)
        value__to__source_submod = __find__TupleIndex_values__generated_from_a_single_acc(root)

        intersection_values = set(value__to__source_submod.keys()).intersection(value__to__target_submods.keys())

        value_path = dict()
        for value in intersection_values:
            value_path[value] = (value__to__source_submod[value], value__to__target_submods[value])

        path_values = dict()
        for value, path in value_path.items():
            if path not in path_values:
                path_values[path] = set()
            path_values[path].add(value)
        # 此时path_values中，每个value既由source submod生成，又多个target submod共同使用

        merge_change = None
        for (source_submod, target_submods), values in path_values.items():
            # 而后我们寻找同时合并target submods输入和source submod输出的机会。
            # 找到那些由source submod的输出直接经过一层TupleIndex到达target submods的value
            source_submod_out_value = source_submod.outputsAt(0)
            temp_target_inputs = list(target_submods[0].inputs())
            vs = [v for v in values
                  if v.node().kind() == "prim::TupleIndex" and v.node().inputsAt(0) == source_submod_out_value and v in temp_target_inputs]
            if 1 < len(vs):
                merge_change = ((source_submod, target_submods), vs)
                break

        return merge_change
    
    merge_change = find_reduce_chance()
    while merge_change is not None:
        (source_submod_calling_node, target_submod_calling_nodes), merged_values = merge_change


        merged_tuple_idx__to__out_idx = list(set([v.node().inputsAt(1).node().i("value") for v in merged_values]))

        # modify the output of source submod
        source_submod = submods[source_submod_calling_node.s("target")]
        out_node = source_submod.return_node().inputsAt(0).node()
        source_submod.setInsertPoint(out_node)

        new_merged_node = source_submod.create("prim::TupleConstruct", [out_node.inputsAt(out_idx) for out_idx in merged_tuple_idx__to__out_idx], 1)
        source_submod.insertNode(new_merged_node)
        new_merged_value = new_merged_node.outputsAt(0)
        new_merged_value.setType(torch.TupleType([out_node.inputsAt(out_idx).type() for out_idx in merged_tuple_idx__to__out_idx]))

        out_node.addInput(new_merged_value)
        out_node.outputsAt(0).setType(torch.TupleType([v.type() for v in out_node.inputs()]))


        # modify the output of source submod calling node in root
        source_value_in_root = source_submod_calling_node.outputsAt(0)
        source_value_in_root.setTypeAs(out_node.outputsAt(0))

        root.setInsertPoint(source_submod_calling_node)
        idx_value = root.insertConstant(out_node.inputsSize() - 1)
        new_merged_node_in_root = root.create("prim::TupleIndex", [source_value_in_root, idx_value], 1)
        new_merged_node_in_root.insertAfter(source_submod_calling_node)

        new_merged_value_in_root = new_merged_node_in_root.outputsAt(0)
        new_merged_value_in_root.setTypeAs(new_merged_value)

        none_value = root.insertConstant(None)

        for target_submod_calling_node in target_submod_calling_nodes:
            # 然后修改root中的target submod调用节点
            out_idx__to__in_idxes = dict()
            for in_idx, in_value in enumerate(target_submod_calling_node.inputs()):
                if in_value not in merged_values:
                    continue
                out_idx = in_value.node().inputsAt(1).node().i("value")

                # 这里又牵扯到一个实参对应多个形参的问题
                if out_idx not in out_idx__to__in_idxes:
                    out_idx__to__in_idxes[out_idx] = set()
                out_idx__to__in_idxes[out_idx].add(in_idx)
            merged_tuple_idx__to__in_idxes = [out_idx__to__in_idxes[out_idx] for _, out_idx in enumerate(merged_tuple_idx__to__out_idx)]

            for in_idxes in merged_tuple_idx__to__in_idxes:
                for in_idx in in_idxes:
                    target_submod_calling_node.replaceInput(in_idx, none_value)
            target_submod_calling_node.addInput(new_merged_value_in_root)


            # 最后修改target submod的输入
            submod = submods[target_submod_calling_node.s("target")]
            new_merged_value_in_target = submod.addInput()
            new_merged_value_in_target.setTypeAs(new_merged_value_in_root)
            dummy_node_for_inserting = submod.create("prim::Constant", [], 1)
            dummy_node_for_inserting.outputsAt(0).setType(torch.NoneType.get())
            submod.prependNode(dummy_node_for_inserting)
            submod.setInsertPoint(dummy_node_for_inserting)
            
            input_values__in_target_submod = list(submod.inputs())
            for i, in_idxes in enumerate(merged_tuple_idx__to__in_idxes):
                new_in_node = submod.create("prim::TupleIndex", [new_merged_value_in_target, submod.insertConstant(i)], 1)
                submod.insertNode(new_in_node)
                for in_idx in in_idxes:
                    orig_in_value = input_values__in_target_submod[in_idx]
                    new_in_value = new_in_node.outputsAt(0)
                    new_in_value.setTypeAs(orig_in_value)
                    orig_in_value.replaceAllUsesWith(new_in_value)

        merge_change = find_reduce_chance()

    return root, submods



# 消除掉因前面处理，多余的NoneType
# %5542 : (NoneType, NoneType, NoneType, NoneType, NoneType, (Float(32000, 4096), Float(4096, 4096), Float(4096), Float(4096, 4096), Float(4096, 4096))) = prim::GCG_Call_Submod[target="submod_bw0"](%311, %primals_293, %merged_state.1, %5607, %5622, %5640, %5670)
def eliminate_redundant_NoneType(root: torch.Graph, submods: Dict):
    for node in root.nodes():
        if node.kind() == "prim::GCG_Call_Submod" and node.output().type().kind() == "TupleType":
            is_NoneType = [type.kind() == "NoneType" for type in node.output().type().elements()]
            has_NoneType = any(is_NoneType)
            if not has_NoneType:
                continue
            users = [use.user for use in node.output().uses()]
            is_user_TupleIndex = [user.kind() == "prim::TupleIndex" for user in users]
            all_users_TupleIndex = all(is_user_TupleIndex)
            if not all_users_TupleIndex:
                continue
            const_index = [use.user.inputsAt(1).node().kind() == "prim::Constant" for use in node.output().uses()]
            all_tuple_index_user__has__const_index = all(const_index)
            if not all_tuple_index_user__has__const_index:
                continue

            # 到了这里，说明这个value的user的tuple index都是常数，这样我们就能把NoneType去掉，然后调整tuple index
            while has_NoneType:
                first_NoneType_index = is_NoneType.index(True)


                # 先把 submod 里的输出改了，对应的NoneType就别输出了
                submod = submods[node.s("target")]
                submod_output_value = next(iter(submod.outputs()))
                tuple_type_elems = submod_output_value.type().elements()
                tuple_type_elems.pop(first_NoneType_index)
                tuple_type = torch.TupleType(tuple_type_elems)
                submod_output_value.setType(tuple_type)

                submod_output_node = submod_output_value.node()
                assert submod_output_node.kind() == "prim::TupleConstruct"
                submod_output_node.removeInput(first_NoneType_index)


                # 然后改 root 计算图
                node.output().setType(tuple_type)
                root.setInsertPoint(next(iter(root.nodes())))
                for user in users:
                    idx = user.inputsAt(1).node().i("value")
                    if idx < first_NoneType_index:
                        pass
                    elif idx == first_NoneType_index:
                        none = root.insertConstant()
                        user.output().replaceAllUsesWith(none)
                    else: # first_NoneType_index < idx
                        new_idx = root.insertConstant(idx - 1)
                        user.replaceInput(1, new_idx)

                pass
                is_NoneType.pop(first_NoneType_index)
                has_NoneType = any(is_NoneType)
                pass
        pass
    return root, submods



# 当TupleType中的NoneType都去掉完了，可能会出现下面的情况
# %91 : ((Float(20, 1, 4, 4), Float(20))) = prim::GCG_Call_Submod[target="submod_bw0"](%75, %20, %33, %primals_9, %merged_state.1)
def eliminate_useless_TupleType(root: torch.Graph, submods: Dict):
    one_more_time = False # 执行一次这个函数，去掉一层TupleType；如果有嵌套的TupleType，就再来一次

    for node in root.nodes():
        if node.kind() == "prim::GCG_Call_Submod":
            if node.output().type().kind() == "TupleType" and 1 == len(node.output().type().elements()):
                users = [use.user for use in node.output().uses()]
                is_user_TupleIndex = [user.kind() == "prim::TupleIndex" for user in users]
                all_users_TupleIndex = all(is_user_TupleIndex)
                if not all_users_TupleIndex:
                    continue
                one_more_time = True

                # 先把 submod 里的输出改了，去掉一层TupleConstruct
                submod = submods[node.s("target")]
                submod_output_value = next(iter(submod.outputs()))
                submod_output_node = submod_output_value.node()
                assert submod_output_node.kind() == "prim::TupleConstruct"
                submod_new_output_value = submod_output_node.inputsAt(0)
                submod.eraseOutput(0);
                submod.registerOutput(submod_new_output_value);
                
                # 然后改 root 计算图
                node.output().setTypeAs(next(iter(submod.outputs())))
                for user in users:
                    user.output().replaceAllUsesWith(node.output())

                pass
            pass
        pass

    if one_more_time:
        return eliminate_useless_TupleType(root, submods)
    return root, submods


if __name__ == "__main__":
    import graph_export
    graph_pack = graph_export.read_from_no_states("llama2_grads_TS")
    ts_mod = torch.parse_ir(graph_pack["ts_fw_root"])
    ts_submods = {}
    for submod_name, submod_str in graph_pack["ts_fw_submods"].items():
        ts_submods[submod_name] = torch.parse_ir(submod_str)
    
    
    eliminate_useless_TupleType(ts_mod, ts_submods)
    
    torch._C._jit_pass_dce(ts_mod)
    set_unsed_submod_output__is__None(ts_mod, ts_submods)
    
    pass