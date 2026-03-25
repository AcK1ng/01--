#include <torch/torch.h>
#include <iostream>
#include <memory>

void press_enter() {
    std::string t;
    std::cout << "Press Enter to continue: " << std::endl;
    std::getline(std::cin, t);
}

std::shared_ptr<torch::jit::IValue>
new_4G_tensor() {
  c10::Device device = c10::Device(torch::kCUDA);
  auto options =
      torch::TensorOptions()
      .dtype(torch::kFloat32)
      .layout(torch::kStrided)
      .device(device)
      .requires_grad(false);

  torch::Tensor tensor1 = torch::empty({1 * (1 << 10), 1 * (1 << 10), 1 * (1 << 10)}, options);
  return std::make_shared<torch::jit::IValue>(tensor1);
}

int main() {
  auto tensor1 = new_4G_tensor();
  std::cout << "new 4g" << std::endl;
  press_enter();
  { 
      auto tensor2 = new_4G_tensor();
      std::cout << "new 4g" << std::endl;
      press_enter();
      std::cout << "release 4g" << std::endl;
  }
  press_enter();
  { 
      auto tensor3 = new_4G_tensor();
      std::cout << "new 4g" << std::endl;
      press_enter();
      std::cout << "release 4g" << std::endl;
  }

  auto tensor4 = new_4G_tensor();
  std::cout << "new 4g" << std::endl;
  press_enter();

  auto tensor5 = new_4G_tensor();
  std::cout << "new 4g" << std::endl;
  press_enter();

  return 0;
}
