#ifndef CLUSTER_H
#define CLUSTER_H

#include "Base.h"

#include <memory>

extern std::shared_ptr<Cluster>
GetNativeCUDACPUCluster();

extern std::shared_ptr<Cluster>
GetNativeCANNCPUCluster();

extern std::shared_ptr<Cluster>
GetNativeCPUCluster();


// After ray::init
extern void Init_HelloWorld_RayCluster();
extern void Init_SingleAccGraph_RayCluster();
extern void Init_TransmitTest_RayCluster();
extern void Init_HTTPedMiniGCGMaster_RayCluster(int port);
extern void Init_HTTPedBigLockGCGMaster_RayCluster(int port);


#endif
