import os
from tvm import autotvm
from tvm import auto_scheduler

autotvm_tuning_option = {
    "n_trial": 2000,
    "early_stopping": 800,
    "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=50),
            runner=autotvm.LocalRunner(number=100, repeat=3, timeout=40) 
        )
}

ansor_tuning_option = {
    "n_trial": 5000,
    "builder": auto_scheduler.LocalBuilder(),
    "repeat": 3,
    "timeout": 40,
    "min_repeat_ms": 200,
    "enable_cpu_cache_flush": True,
}



