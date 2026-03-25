#pragma once
#include <iostream>

namespace ors {

enum class DeviceFlag : uint8_t {
  CPU = 0,
  GPU,
  NPU,
  UPBOUND
};


}