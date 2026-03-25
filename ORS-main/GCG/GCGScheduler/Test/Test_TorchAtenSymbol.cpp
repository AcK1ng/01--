#include <iostream>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/api/include/torch/types.h>

#include <sstream>
#include <vector>
#include <c10/core/Layout.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

int
main () {
    const torch::jit::IValue shape = std::vector<int64_t>({3, 4});
    const torch::jit::IValue dtype = at::kDouble;
    const torch::jit::IValue layout = c10::kStrided;
    const torch::jit::IValue device = c10::Device("cuda");
    const torch::jit::IValue none = torch::jit::IValue();

    torch::jit::Stack stack;
    torch::jit::push_one(stack, shape);
    torch::jit::push_one(stack, dtype);
    torch::jit::push_one(stack, layout);
    torch::jit::push_one(stack, device);
    torch::jit::push_one(stack, none);
    torch::jit::push_one(stack, none);

    //const auto symbol = c10::Symbol::fromQualString("aten::zeros");
    const auto symbol = c10::Symbol::fromQualString("aten::mul");
    const auto ops = torch::jit::getAllSortedOperatorsFor(symbol);
    bool runned = 0;
    for (auto op: ops) {
        std::cout << op->schema() << std::endl;
        auto &formals = op->schema().arguments();
        if (formals.size() != stack.size())
            continue;
        try {
            op->getOperation()(stack);
            runned = 1;
        } catch(std::exception e) {
            std::cout << "exec failed: " << e.what() << std::endl;
        }
        if (runned)
            break;
    }
    std::cout << "runned" << runned << std::endl;

    auto res = torch::jit::pop(stack);
    std::cout << res << std::endl;

    return 0;
}
