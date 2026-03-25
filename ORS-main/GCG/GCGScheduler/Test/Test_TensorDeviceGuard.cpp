#include <torch/torch.h>


#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <c10/core/Layout.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
//#include <c10/cuda/CUDAGuard.h>

#include <c10/core/DispatchKeySet.h>

int
main () {
    {
        c10::DispatchKeySet key_set1 = c10::DispatchKeySet(c10::DispatchKey::Meta);
        c10::DispatchKeySet key_set = c10::DispatchKeySet();
        c10::impl::ForceDispatchKeyGuard guard(key_set1, key_set);
        // torch::DeviceGuard device_guard(c10::Device(torch::kMeta));
        torch::Tensor t = torch::empty({1, 1});
        std::cout << t.device() << std::endl;
    }

    {
        // 创建索引张量，表示要提取的嵌入向量的索引
        torch::Tensor indices = torch::tensor({1, 3, 5, 1, 2}, torch::kLong); // 注意索引数据类型为 long
        std::cout << "索引张量:\n" << indices.device() << std::endl;
        
        c10::DispatchKeySet key_set1 = c10::DispatchKeySet(c10::DispatchKey::Meta);
        c10::DispatchKeySet key_set = c10::DispatchKeySet();
        c10::impl::ForceDispatchKeyGuard guard(key_set1, key_set);
        // 假设我们有一个权重矩阵，形状为 (num_embeddings, embedding_dim)
        // 例如：6 个嵌入向量，每个维度为 4
        torch::Tensor weight = torch::randn({6, 4}); // 随机初始化权重
        std::cout << "权重矩阵:\n" << weight.device() << std::endl;
        std::cout << weight.dim() << std::endl;
        std::cout << weight.scalar_type() << std::endl;


    }

    return 0;
}
