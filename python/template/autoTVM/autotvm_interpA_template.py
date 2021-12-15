import re
import tvm
from tvm import te
from tvm import autotvm
from utils.tune_task import android_autotvm_tune_task, linux_autotvm_tune_task

dtype = "float32"

@autotvm.template('cubic_matmul_tvm_interpA')
def gemm_cubic_interpA_kernel_fuseK_template(dim, name, target):
    M = dim[0]
    N = dim[1]
    K = dim[2]
    global cfg
    cfg = autotvm.get_config()
    cfg.define_split('tile_x', M, num_outputs=3)
    cfg.define_split('tile_y', N, num_outputs=3)
    cfg.define_split('tile_k', K, num_outputs=3)
    mc = cfg['tile_x'].size[-1]
    kc = cfg['tile_k'].size[-1]
    nc = cfg['tile_y'].size[-1]
    W = 4
    weights_num = W
    Coeff = te.placeholder((M, 4), dtype=dtype, name='coeffcient')

    #============pack===============
    WE = te.placeholder((W, K, N), name='Weights')
    stride = nc
    packWE = te.compute((N // stride, K * W, stride), lambda x, y, z:WE[(y // K) % W, y % K, z + (stride * x)] , name="B")
    s_packw = te.create_schedule(packWE.op)
    global packFunc
    packFunc = tvm.build(s_packw, [WE, packWE], target=target, name="{}_pack".format(name))   
    #===============================
    k = te.reduce_axis((0, W * K), name="k")
    A = te.placeholder((M, K), name="A")
    Coeff = te.placeholder((M, W), dtype=dtype, name='coeffcient')
    B = te.placeholder((N // stride, K * W, stride), name="B")
    interpA = te.compute((M, W * K), lambda x, k : Coeff[x, (k // K) % W] * A[x, (k % K)], name="interpA")
    #interpolation A
    C = te.compute((M, N), lambda x, y:te.sum(interpA[x, k] * B[y // stride, k, y % stride], axis=k), name='C')
    s = te.create_schedule(C.op)
    (x, y) = s[C].op.axis
    (k, ) = s[C].op.reduce_axis
    xt, xo, xi = cfg['tile_x'].apply(s, C, x)
    yt, yo, yi = cfg['tile_y'].apply(s, C, y)
    kt, ko, ki = cfg['tile_k'].apply(s, C, k)
    s[C].reorder(xt, xo, kt, yt, yo, ko, ki, xi, yi)

    # # without xsmm 
    s[interpA].compute_at(s[C], yi)
    cfg.define_reorder('reorder', [xt, xo, kt, yt, yo, ko, ki, xi, yi], 'all')
    cfg.define_annotate('annotate', [xt, xo, kt, yt, yo, ko, ki, xi, yi], 'try_unroll_vec')
    cfg['reorder'].apply(s, C, [xt, xo, kt, yt, yo, ko, ki, xi, yi])
    cfg['annotate'].apply(s, C, [xt, xo, kt, yt, yo, ko, ki, xi, yi])
    s[interpA].compute_inline()
    
    return s, [A, B, Coeff, C]

def autotune_interp_A_kernel(dim, target, name='defaultAutotune', use_tune=False, print_lower=True):
    log_dir = './logs/logs_interpA_autoTVM/tune_M{}_N{}_K{}.log'.format(dim[0], dim[1], dim[2])

    if use_tune:
        task = autotvm.task.create('cubic_matmul_tvm_interpA', args=(dim, name, target), target=target)
        if re.search(r"aarch64-linux-android", target):
            android_autotvm_tune_task(task, log_dir)
        elif re.search(r"skylake-avx512", target) or re.search(r"core-avx2", target):
            linux_autotvm_tune_task(task, log_dir)
    
    with autotvm.apply_history_best(log_dir):
        with tvm.target.Target(target):
            s, args = gemm_cubic_interpA_kernel_fuseK_template(dim, name, target)
            cubic_gemm_func = tvm.build(s, args, name=name)
            nc = cfg['tile_y'].size[-1]
            kc = cfg['tile_k'].size[-1]
            if print_lower:
                print("========= autotune interpA ===========")
                print(tvm.lower(s, args, simple_mode=True))

    return [cubic_gemm_func, packFunc, nc, kc]