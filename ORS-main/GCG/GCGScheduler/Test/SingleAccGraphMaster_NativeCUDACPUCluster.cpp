#include "Cluster/Cluster.h"
#include "Master/Master.h"
#include <thread>
#include <chrono>

int
main () {
  ClusterSys simple_simulator(GetNativeCUDACPUCluster(), GetSingleAccGraphMaster());
  simple_simulator.StartUp();

  while (1) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
  return 0;
}
