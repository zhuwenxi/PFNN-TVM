import re
import tvm
from tvm import te
from tvm import auto_scheduler
from utils.tune_task import android_ansor_tune_task, linux_ansor_tune_task

dtype = "float32"

@auto_scheduler.register_workload
def ansor_gemm_cubic_interpA_kernel_fuseK(dim, name, target):
    M = dim[0]
    N = dim[1]
    K = dim[2]
    W =  4 # Four copies of weights, constant.
    nc = 16
    kc = K
    stride = nc

    Coeff = te.placeholder((M, 4), dtype=dtype, name='coeffcient')

    k = te.reduce_axis((0, W * K), name="k")
    A = te.placeholder((M, K), name="A")
    Coeff = te.placeholder((M, W), dtype=dtype, name='coeffcient')
    B = te.placeholder((N // stride, K * W, stride), name="B")
    interpA = te.compute((M, W * K), lambda x, k : Coeff[x, (k // K) % W] * A[x, (k % K)], name="interpA")
    C = te.compute((M, N), lambda x, y:te.sum(interpA[x, k] * B[y // stride, k, y % stride], axis=k), name='C')
    return [A, B, Coeff, C]


def ansor_tune_interp_A_kernel(dim, target, name='defaultAnsorAutotune', use_tune=False, print_lower=True):
    task = auto_scheduler.SearchTask(func=ansor_gemm_cubic_interpA_kernel_fuseK, args=(dim, name, target), target=target)
    log_dir = './logs/logs_interpA_ansor/tune_M{}_N{}_K{}.log'.format(dim[0], dim[1], dim[2])
    if use_tune:
        if re.search(r"aarch64-linux-android", target):
            android_ansor_tune_task(task, log_dir)
        elif re.search(r"skylake-avx512", target) or re.search(r"core-avx2", target):
            linux_ansor_tune_task(task, log_dir)
    
    s, args = task.apply_best(log_dir)
    cubic_gemm_func = tvm.build(s, args, target, name=name)

    M = dim[0]
    N = dim[1]
    K = dim[2]
    W =  4 # Four copies of weights, constant.
    nc = 16
    kc = K
    stride = nc
    #============pack===============
    WE = te.placeholder((W, K, N), name='Weights')
    packWE = te.compute((N // stride, K * W, stride), lambda x, y, z:WE[(y // K) % W, y % K, z + (stride * x)] , name="B")
    s_packw = te.create_schedule(packWE.op)
    packFunc = tvm.build(s_packw, [WE, packWE], target=target, name="{}_pack".format(name))   
    
    if print_lower:
        print("========= ansor interpA ===========")
        print(tvm.lower(s, args, simple_mode=True))
    return [cubic_gemm_func, packFunc, nc, kc]
