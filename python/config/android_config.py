import os
from tvm import autotvm
from tvm import auto_scheduler

tracker_host = "0.0.0.0"
tracker_port = 9190
rpc_key = "android"

if "TVM_TRACKER_HOST" in os.environ: 
    tracker_host = os.environ["TVM_TRACKER_HOST"]
if "TVM_TRACKER_PORT" in os.environ: 
    tracker_port = int(os.environ["TVM_TRACKER_PORT"])
if "TVM_NDK_CC" not in os.environ:
    raise RuntimeError(
    "Require environment variable TVM_NDK_CC" " to be the NDK standalone compiler"
)
cc_compiler = os.environ["TVM_NDK_CC"]

autotvm_tuning_option_rpc = {
    "n_trial": 2000,
    "early_stopping": 800,
    "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="ndk", n_parallel=4, timeout=50),
            runner=autotvm.RPCRunner(key=rpc_key, host=tracker_host, port=tracker_port, number=100, repeat=3, timeout=40) 
        )
}

ansor_tuning_option_rpc = {
    "n_trial": 2000,
    "builder": auto_scheduler.LocalBuilder(build_func="ndk"),
    "repeat": 3,
    "timeout": 40,
    "min_repeat_ms": 200,
    "enable_cpu_cache_flush": True,
}



