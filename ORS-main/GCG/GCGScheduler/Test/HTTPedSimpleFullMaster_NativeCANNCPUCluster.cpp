#include "Cluster/Cluster.h"
#include "Master/Master.h"
#include <thread>
#include <chrono>

int
main () {
  auto clustersys = ClusterSys(GetNativeCANNCPUCluster(), WrapHTTPedMaster(GetSimpleFullMaster(), 6067));
  clustersys.StartUp();

  while (1) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }

  return 0;
}
