import tvm
from tvm import te
from utils.xsmm_utils.tvm_extern_xsmm import intrin_gemm_MxKxN, gemm_MxKxN_impl

dtype = "float32"

def gemm_cubic_interp_A_kernel_fuseK_with_xsmm(dim, target, name='defaultGEMMCubicInterpolation', print_lower=False):
    M = dim[0]
    N = dim[1]
    K = dim[2]
    kc = K
    nc = 16
    mc = M
    W =  4 # Four copies of weights, constant.
    nr = nc # Vectorization length
    stride = nc
    weights_num = W
    WE = te.placeholder((W, K, N), name="Weights")
    packWE = te.compute((N // stride, K * W, stride), lambda x, y, z:WE[(y // K) % W, y % K, z + (stride * x)] , name="B")
    s_packw = te.create_schedule(packWE.op)
    packFunc = tvm.build(s_packw, [WE, packWE], target=target, name="{}_pack".format(name))
    #==========================================================
    k = te.reduce_axis((0, W * K), name="k")
    A = te.placeholder((M, K), name="A")
    Coeff = te.placeholder((M, W), dtype=dtype, name='coeffcient')
    B = te.placeholder((N // stride, K * W, stride), name="B")
    interpA = te.compute((M, W * K), lambda x, k : Coeff[x, (k // K) % W] * A[x, (k % K)], name="interpA")
    #interpolation A
    C = te.compute((M, N), lambda x, y:te.sum(interpA[x, k] * B[y // stride, k, y % stride], axis=k), name='C')

    s = te.create_schedule(C.op)
    (x, y) = s[C].op.axis
    (reduce_k, ) = s[C].op.reduce_axis
    xo, xi = s[C].split(x, mc)
    ko, ki = s[C].split(reduce_k, kc)
    yt, yu = s[C].split(y, nc)
    yo, yi = s[C].split(yu, stride)
    s[C].reorder(xo, ko, yt, yo, ki, xi, yi)

    # with xsmm
    micro_kernel, uniq_id = intrin_gemm_MxKxN(M, W * K, N, mc, kc, nc)
    s[C].tensorize(ki, micro_kernel)
    s[C].pragma(xo, "import_llvm", gemm_MxKxN_impl(mc, kc, nc, W * K, nc, N, uniq_id))

    func = tvm.build(s, [A, B, Coeff, C], target=target, name=name)
    if print_lower:
        print(tvm.lower(s, [A, B, Coeff, C], simple_mode=True))
    return [func, packFunc, nc, kc]
