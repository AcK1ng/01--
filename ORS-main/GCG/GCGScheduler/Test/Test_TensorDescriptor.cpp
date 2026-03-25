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

struct ElemDescriptor {
  int is_tensor;
  at::IntArrayRef shape;
  c10::TensorOptions options;
};
struct TupleDescriptor {
  int tuple_wrapped__when_nr_elem_is_1;
  std::vector<ElemDescriptor> elem_descriptors;
};

inline struct ElemDescriptor
IValue__to__ElemDescriptor(c10::IValue &ivalue) {
  struct ElemDescriptor ret;
  if (ivalue.isTensor()) {
    ret.is_tensor = 1;
    ret.shape = ivalue.toTensor().sizes(),
    ret.options.dtype(ivalue.toTensor().dtype())
      .layout(torch::kStrided)
      .requires_grad(false);

    auto scalartype = ivalue.toTensor().dtype().toScalarType();
    int scalartype__int = (int)scalartype;

    std::cout << scalartype << std::endl;
    std::cout << scalartype__int << std::endl;
    std::cout << c10::ScalarType(scalartype__int) << std::endl;
  } else if (ivalue.isScalar()) {
    assert(0);
  } else
    assert(0);
  return ret;
}

struct TupleDescriptor
IValue__to__TupleDescriptor(c10::IValue &ivalue) {
  struct TupleDescriptor ret;
  if (ivalue.isTuple()) {
    assert(0);
  } else {
    ret.tuple_wrapped__when_nr_elem_is_1 = 0;
    ret.elem_descriptors.push_back(IValue__to__ElemDescriptor(ivalue));
  }
  return ret;
}

inline c10::IValue
ElemDescriptor__to__IValue(struct ElemDescriptor &elem_descriptor) {
  if (elem_descriptor.is_tensor) 
    return torch::empty(elem_descriptor.shape, elem_descriptor.options);
  else
    assert(0);
}

c10::IValue
TupleDescriptor__to__IValue(struct TupleDescriptor tuple_descriptor) {
  if (tuple_descriptor.tuple_wrapped__when_nr_elem_is_1) {
    assert(0);
  } else {
    return ElemDescriptor__to__IValue(tuple_descriptor.elem_descriptors[0]);
  }
}

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

    auto tuple_descriptor = IValue__to__TupleDescriptor(res);
    auto empty_tensor = TupleDescriptor__to__IValue(tuple_descriptor);
    std::cout << empty_tensor << std::endl;

    return 0;
}
