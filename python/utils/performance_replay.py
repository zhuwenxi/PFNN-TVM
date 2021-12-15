import re
import tvm
import numpy as np
from tvm import rpc
from tvm.contrib import utils, ndk


def schedule_performance_replay_interpB(dim, target, gemmFunc, packFunc, nc, kc):
    if re.search(r"aarch64-linux-android", target):
        from config.android_config import tracker_host, tracker_port, rpc_key
        temp = utils.tempdir()
        path_dso_cpu = temp.relpath("cpu_lib.so")
        gemmFunc.export_library(path_dso_cpu, ndk.create_shared)
        path_prepack_dso_cpu = temp.relpath("cpu_pack_lib.so")
        packFunc.export_library(path_prepack_dso_cpu, ndk.create_shared)

        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(rpc_key, priority=0, session_timeout=500)
        config_func = remote.get_function("runtime.config_threadpool")
        config_func(1, 1)

        remote.upload(path_dso_cpu)
        gemmFunc = remote.load_module("cpu_lib.so")
        remote.upload(path_prepack_dso_cpu)
        packFunc = remote.load_module("cpu_pack_lib.so")
        ctx = remote.cpu(0)
    elif re.search(r"skylake-avx512", target) or re.search(r"core-avx2", target):
        ctx = tvm.device(target, 0)

    dtype = "float32"
    stride, kc = nc, kc

    input_tensor = tvm.nd.array(np.random.rand(dim[0], dim[2]).astype(dtype), ctx)
    input_weight = tvm.nd.array(np.random.rand(4, dim[2], dim[1]).astype(dtype), ctx)
    coeff = tvm.nd.array(np.random.rand(dim[0], 4).astype(dtype), ctx)
    input_weight_pack = tvm.nd.array(np.random.rand(dim[2] // kc, dim[1] // stride, kc, 4, stride).astype(dtype), ctx)
    output_tensor = tvm.nd.array(np.zeros((dim[0], dim[1]), dtype=dtype), ctx)
    packFunc(input_weight, input_weight_pack)
    gemmFunc(input_tensor, input_weight_pack, coeff, output_tensor)
    evaluator = gemmFunc.time_evaluator(gemmFunc.entry_name, ctx, number=1000, repeat=10)
    mean_time  = evaluator(input_tensor, input_weight_pack, coeff, output_tensor).mean
    gflops = 10 * dim[0] * dim[1] * dim[2] * 1e-9 / mean_time
    print("shape : ({},{},{}), Cost Time: {}ms, GFlops : {}".format(dim[0], dim[1], dim[2], mean_time * 1000, gflops))



def schedule_performance_replay_interpA(dim, target, gemmFunc, packFunc, nc, kc):
    if re.search(r"aarch64-linux-android", target):
        from config.android_config import tracker_host, tracker_port, rpc_key
        temp = utils.tempdir()
        path_dso_cpu = temp.relpath("cpu_lib.so")
        gemmFunc.export_library(path_dso_cpu, ndk.create_shared)
        path_prepack_dso_cpu = temp.relpath("cpu_pack_lib.so")
        packFunc.export_library(path_prepack_dso_cpu, ndk.create_shared)

        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(rpc_key, priority=0, session_timeout=500)
        config_func = remote.get_function("runtime.config_threadpool")
        config_func(1, 1)

        remote.upload(path_dso_cpu)
        gemmFunc = remote.load_module("cpu_lib.so")
        remote.upload(path_prepack_dso_cpu)
        packFunc = remote.load_module("cpu_pack_lib.so")
        ctx = remote.cpu(0)
    elif re.search(r"skylake-avx512", target) or re.search(r"core-avx2", target):
        ctx = tvm.device(target, 0)

    dtype = "float32"
    stride, kc = nc, kc
    input_tensor = tvm.nd.array(np.random.rand(dim[0], dim[2]).astype(dtype), ctx)
    input_weight = tvm.nd.array(np.random.rand(4, dim[2], dim[1]).astype(dtype), ctx)
    coeff = tvm.nd.array(np.random.rand(dim[0], 4).astype(dtype), ctx)
    input_weight_pack = tvm.nd.array(np.random.rand(dim[1] // stride, dim[2] * 4, stride).astype(dtype), ctx)
    output_tensor = tvm.nd.array(np.zeros((dim[0], dim[1]), dtype=dtype), ctx)
    packFunc(input_weight, input_weight_pack)
    gemmFunc(input_tensor, input_weight_pack, coeff, output_tensor)
    evaluator = gemmFunc.time_evaluator(gemmFunc.entry_name, ctx, number=1000, repeat=10) 
    mean_time  = evaluator(input_tensor, input_weight_pack, coeff, output_tensor).mean
    gflops = (4 * dim[0] * dim[2] + 8 * dim[0] * dim[1] * dim[2]) * 1e-9 / mean_time
    print("shape : ({},{},{}), Cost Time: {}ms, GFlops : {}".format(dim[0], dim[1], dim[2], mean_time * 1000, gflops))
