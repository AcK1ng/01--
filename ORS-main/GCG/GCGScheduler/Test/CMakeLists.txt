

add_executable(Test_TorchRuntime Test_TorchRuntime.cpp)
target_link_libraries(Test_TorchRuntime
        "${TORCH_LIBRARIES}"
        pthread)

add_executable(Test_TensorDescriptor Test_TensorDescriptor.cpp)
target_link_libraries(Test_TensorDescriptor
        "${TORCH_LIBRARIES}"
        pthread)

add_executable(Test_TensorDeviceGuard Test_TensorDeviceGuard.cpp)
target_link_libraries(Test_TensorDeviceGuard
        "${TORCH_LIBRARIES}"
        pthread)

add_executable(Test_TorchAtenSymbol Test_TorchAtenSymbol.cpp)
target_link_libraries(Test_TorchAtenSymbol
        "${TORCH_LIBRARIES}"
        pthread)

add_executable(Test_TorchGraph Test_TorchGraph.cpp)
target_link_libraries(Test_TorchGraph
        "${TORCH_LIBRARIES}"
        pthread)

#add_executable(Test_TorchDist_node0 Test_TorchDist_node0.cpp)
#target_link_libraries(Test_TorchDist_node0
#        "${TORCH_LIBRARIES}"
#        pthread)
#add_executable(Test_TorchDist_node1 Test_TorchDist_node1.cpp)
#target_link_libraries(Test_TorchDist_node1
#        "${TORCH_LIBRARIES}"
#        pthread)
#add_executable(Test_TorchDist_node2 Test_TorchDist_node2.cpp)
#target_link_libraries(Test_TorchDist_node2
#        "${TORCH_LIBRARIES}"
#        pthread)


add_executable(TransmitTestMaster_NativeCUDACPUCluster TransmitTestMaster_NativeCUDACPUCluster.cpp)
target_link_libraries(TransmitTestMaster_NativeCUDACPUCluster
        GCGScheduler)

add_executable(Test_Json Test_Json.cpp)
target_link_libraries(Test_Json GCGScheduler)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(PythonGCG PythonGCG.cpp)
target_link_libraries(PythonGCG PRIVATE GCGScheduler)


add_executable(HTTPedSimpleFullMaster_NativeCPUCluster HTTPedSimpleFullMaster_NativeCPUCluster.cpp)
target_link_libraries(HTTPedSimpleFullMaster_NativeCPUCluster GCGScheduler)

add_executable(HTTPedSimpleFullMaster_NativeCUDACPUCluster HTTPedSimpleFullMaster_NativeCUDACPUCluster.cpp)
target_link_libraries(HTTPedSimpleFullMaster_NativeCUDACPUCluster GCGScheduler)

add_executable(HTTPedSimpleFullMaster_NativeCANNCPUCluster HTTPedSimpleFullMaster_NativeCANNCPUCluster.cpp)
target_link_libraries(HTTPedSimpleFullMaster_NativeCANNCPUCluster GCGScheduler)

add_executable(Test_priority Test_priority.cpp)
target_link_libraries(Test_priority GCGScheduler)

add_executable(Test_task_profiling Test_task_profiling.cpp)
target_link_libraries(Test_task_profiling GCGScheduler)

add_executable(Test_transmit_profiling Test_transmit_profiling.cpp)
target_link_libraries(Test_transmit_profiling GCGScheduler)

add_executable(Test_StructKey Test_StructKey.cpp)
target_link_libraries(Test_StructKey GCGScheduler)

add_executable(Test_ShapredPtr_Tensor Test_ShapredPtr_Tensor.cpp)
target_link_libraries(Test_ShapredPtr_Tensor GCGScheduler)

add_executable(HTTPedFullClusterSimulator HTTPedFullClusterSimulator.cpp)
target_link_libraries(HTTPedFullClusterSimulator GCGScheduler)

add_executable(SingleAccGraphMaster_NativeCUDACPUCluster SingleAccGraphMaster_NativeCUDACPUCluster.cpp)
target_link_libraries(SingleAccGraphMaster_NativeCUDACPUCluster
        GCGScheduler)
