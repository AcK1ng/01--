import torch

import torch

def basic_tuple_index_example(x):
    # 创建包含多个张量的元组
    tuple_data = (x * 2, x + 1, x - 1)

    # 这些操作会生成 prim::TupleIndex
    first_element = tuple_data[0]    # prim::TupleIndex(tuple_data, 0)
    second_element = tuple_data[1]   # prim::TupleIndex(tuple_data, 1)
    third_element = tuple_data[2]    # prim::TupleIndex(tuple_data, 2)

    return first_element + second_element + third_element


# 脚本化函数
scripted_add = torch.jit.script(basic_tuple_index_example)

graph = scripted_add.graph

print(graph)
