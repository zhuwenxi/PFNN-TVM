import re
import tvm
from tvm import te
from tvm import auto_scheduler
from utils.tune_task import android_ansor_tune_task, linux_ansor_tune_task

dtype = "float32"

@auto_scheduler.register_workload
def ansor_gemm_cubic_interpB_kernel(dim, name, target):
    M = dim[0]
    N = dim[1]
    K = dim[2]
    W =  4 # Four copies of weights, constant.
    nc = 16
    kc = K
    stride = nc

    Coeff = te.placeholder((M, 4), dtype=dtype, name='coeffcient')

    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K // kc, N // stride, kc, W, stride), name="B")
    interpB = te.compute((M, K // kc, N / stride, kc, stride), lambda x, xo, y, xi, z:
        Coeff[x, 0] * B[xo, y, xi, 0, z] + Coeff[x, 1] * B[xo, y, xi, 1, z] + Coeff[x, 2] * B[xo, y, xi, 2, z] + Coeff[x, 3] * B[xo, y, xi, 3, z], 
        name="interpB")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * interpB[x, k // kc, y // nc, tvm.tir.indexmod(k, kc), tvm.tir.indexmod(y, nc)], axis=k), name='C')
    return [A, B, Coeff, C]


def ansor_tune_interp_B_kernel(dim, target, name='defaultAnsorAutotune', use_tune=False, print_lower=True):
    task = auto_scheduler.SearchTask(func=ansor_gemm_cubic_interpB_kernel, args=(dim, name, target), target=target)
    log_dir = './logs/logs_interpB_ansor/tune_M{}_N{}_K{}.log'.format(dim[0], dim[1], dim[2])
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
    WE = te.placeholder((W, K, N), name="Weights")
    packWE = te.compute((K // kc, N // stride, kc, W, stride), lambda xo, y, xi, w, z: WE[w, xo * kc + xi, y * stride + z], name="B")
    s_packw = te.create_schedule(packWE.op)
    pack_ko, pack_no, pack_kc, pack_w, pack_nc = s_packw[packWE].op.axis
    s_packw[packWE].vectorize(pack_nc)
    #print("===============pack weight=======================")
    #print(tvm.lower(s_packw, [WE, packWE], simple_mode=False))
    packFunc = tvm.build(s_packw, [WE, packWE], target=target, name="{}_pack".format(name))

    if print_lower:
        print("========= ansor interpB ===========")
        print(tvm.lower(s, args, simple_mode=True))
    return [cubic_gemm_func, packFunc, nc, kc]
