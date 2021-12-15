cmake_minimum_required(VERSION 3.2)

project(pfnn_tvm_benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")

include_directories(third_party/eigen3)
include_directories(third_party/tvm_deploy/include)
include_directories(src/pfnn_tvm_src)
add_library(tvm_runtime_pack STATIC third_party/tvm_deploy/tvm_runtime_pack.cc)

aux_source_directory(src/pfnn_tvm_src/ INFER_SRCS)
message(STATUS "Found inference source files: ${INFER_SRCS}")
add_library(infer_pack STATIC ${INFER_SRCS})
target_compile_options(infer_pack PUBLIC -Ofast -g)
if (${EIGEN_INTERP} STREQUAL "A")
    target_compile_definitions(infer_pack PUBLIC -DEIGEN_INFER_BATCH_MODE)
endif ()
if (${TVM_INTERP} STREQUAL "A")
    target_compile_definitions(infer_pack PUBLIC -DTVM_INFER_BATCH_MODE)
endif ()
target_compile_definitions(infer_pack PUBLIC -DPFNN_USE_SYSTEM_LIB)

if (${PLATFORM} STREQUAL "android")
    find_library(tvm_mod_lib tvminfer_android_aarch64 "./build")
elseif (${PLATFORM} STREQUAL "linux")
    find_library(tvm_mod_lib tvminfer_linux_x86_64 "./build")
endif ()
message(STATUS "Found tvm mod lib ${tvm_mod_lib}")

add_executable(pfnn_benchmark src/benchmark.cpp build/dummy_stub.cc)
target_link_libraries(pfnn_benchmark PUBLIC
    ${tvm_mod_lib}
    infer_pack
    tvm_runtime_pack
    -static-libstdc++
    -pthread
)


