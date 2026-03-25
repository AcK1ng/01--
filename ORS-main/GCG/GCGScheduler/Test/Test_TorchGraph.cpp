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
    torch::jit::Stack stack;

    std::string graph_str =
"\
graph():\n\
  %12 : bool? = prim::Constant()\n\
  %10 : Device? = prim::Constant()\n\
  %6 : int? = prim::Constant()\n\
  %4 : int[] = prim::Constant[value=[3,3]]()\n\
  %rv.1 : Tensor = aten::zeros(%4, %6, %6, %10, %12)\n\
  return (%rv.1)\n\
";


    auto graph = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph_str, graph.get());
    torch::jit::Code code(graph, "<on-demand-func>");
    torch::jit::InterpreterState(code).run(stack);

    auto res = torch::jit::pop(stack);
    std::cout << res << std::endl;

    return 0;
}
