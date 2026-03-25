#include <torch/torch.h>
#include <iostream>
#include <vector>

int main() {
  torch::Tensor tensor = torch::tensor({1.0, 1.0, 1.0, 1.0}, torch::kFloat);
  auto shape = tensor.sizes();

  c10::Device device = c10::Device(torch::kCPU);
  auto options =
      torch::TensorOptions()
      .dtype(torch::kFloat32)
      .layout(torch::kStrided)
      .device(device)
      .requires_grad(false);

  torch::Tensor tensor1 = torch::empty(shape, options);
  std::cout << shape << std::endl;
  std::cout << tensor1 << std::endl;
}
