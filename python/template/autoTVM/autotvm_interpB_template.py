import re
import tvm
from tvm import te
from tvm import autotvm
from utils.tune_task import android_autotvm_tune_task, linux_autotvm_tune_task

dtype = "float32"

@autotvm.template('cubic_matmul_tvm_interpB')
def gemm_cubic_interpB_kernel_template(dim, name, target):
    M = dim[0]
    N = dim[1]
    K = dim[2]
    global cfg
    cfg = autotvm.get_config()
    cfg.define_split('tile_y', N, num_outputs=3)
    cfg.define_split('tile_k', K, num_outputs=3)
    kc = cfg['tile_k'].size[-1]
    nc = cfg['tile_y'].size[-1]
    W =   4 # Four copies of weights, constant.
    nr = nc # Vectorization length
    stride = nc
    weights_num = W
    Coeff = te.placeholder((M, 4), dtype=dtype, name='coeffcient')
    #============pack=============
    WE = te.placeholder((W, K, N), name="Weights")
    kc = cfg['tile_k'].size[-1]
    weights_num = W
    stride = cfg['tile_y'].size[-1] 
    packWE = te.compute((K // kc, N // stride, kc, weights_num, stride), lambda xo, y, xi, w, z: WE[w, xo * kc + xi, y * stride + z], name="B")
    s_packw = te.create_schedule(packWE.op)
    pack_ko, pack_no, pack_kc, pack_w, pack_nc = s_packw[packWE].op.axis
    s_packw[packWE].vectorize(pack_nc)
    #print("===============pack weight=======================")
    #print(tvm.lower(s_packw, [WE, packWE], simple_mode=False))
    global packFunc
    packFunc = tvm.build(s_packw, [WE, packWE], target=target, name="{}_pack".format(name))
    #==============================
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K // kc, N // stride, kc, weights_num, stride), name="B")
    interpB = te.compute((M, K // kc, N / stride, kc, stride), lambda x, xo, y, xi, z:
        Coeff[x, 0] * B[xo, y, xi, 0, z] + Coeff[x, 1] * B[xo, y, xi, 1, z] + Coeff[x, 2] * B[xo, y, xi, 2, z] + Coeff[x, 3] * B[xo, y, xi, 3, z], 
        name="interpB")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * interpB[x, k // kc, y // nc, tvm.tir.indexmod(k, kc), tvm.tir.indexmod(y, nc)], axis=k), name='C')
    s = te.create_schedule(C.op)
    (k, ) = s[C].op.reduce_axis
    (x, y) = s[C].op.axis
    yt, yo, yi = cfg['tile_y'].apply(s, C, y)
    kt, ko, ki = cfg['tile_k'].apply(s, C, k)
    s[C].reorder(yo, kt, ko, yt, ki, x, yi)
    cfg.define_reorder('reorder', [yo, kt, ko, yt, ki, x, yi], "all")
    cfg.define_annotate('annotate', [yo, kt, ko, yt, ki, x, yi] , "try_unroll_vec")

    cfg['reorder'].apply(s, C, [yo, kt, ko, yt, ki, x, yi])
    cfg['annotate'].apply(s, C, [yo, kt, ko, yt, ki, x, yi])
    s[interpB].compute_at(s[C], yi)

    s[interpB].compute_inline()
    return s, [A, B, Coeff, C]

def autotune_interp_B_kernel(dim, target, name='defaultAutotune', use_tune=False, print_lower=True):
    log_dir = './logs/logs_interpB_autoTVM/tune_M{}_N{}_K{}.log'.format(dim[0], dim[1], dim[2])

    if use_tune:
        task = autotvm.task.create('cubic_matmul_tvm_interpB', args=(dim, name, target), target=target)
        if re.search(r"aarch64-linux-android", target):
            android_autotvm_tune_task(task, log_dir)
        elif re.search(r"skylake-avx512", target) or re.search(r"core-avx2", target):
            linux_autotvm_tune_task(task, log_dir)

    with autotvm.apply_history_best(log_dir):
        with tvm.target.Target(target):
            s, args = gemm_cubic_interpB_kernel_template(dim, name, target)
            cubic_gemm_func = tvm.build(s, args, name=name)
            nc = cfg['tile_y'].size[-1]
            kc = cfg['tile_k'].size[-1]
            if print_lower:
                print("========= autotune interpB ===========")
                print(tvm.lower(s, args, simple_mode=True))

    return [cubic_gemm_func, packFunc, nc, kc]
