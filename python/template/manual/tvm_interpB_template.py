import tvm
from tvm import te

dtype = "float32"

def gemm_cubic_interp_B_kernel(dim, target, name='defaultGEMMCubicInterpolation', print_lower=True):
    M = dim[0]
    N = dim[1]
    K = dim[2]
    W =   4 # Four copies of weights, constant.
    nc = 16
    kc = K
    nr = nc # Vectorization length
    stride = nc
    weights_num = W
    Coeff = te.placeholder((M, 4), dtype=dtype, name='coeffcient')

    WE = te.placeholder((W, K, N), name="Weights")
    packWE = te.compute((K // kc, N // stride, kc, weights_num, stride), lambda xo, y, xi, w, z: WE[w, xo * kc + xi, y * stride + z], name="B")
    s_packw = te.create_schedule(packWE.op)
    #print(tvm.lower(s_packw, [WE, packWE], simple_mode=False))
    packFunc = tvm.build(s_packw, [WE, packWE], target=target, name="{}_pack".format(name))
    
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K // kc, N // stride, kc, weights_num, stride), name="B")

    interpB = te.compute((M, K // kc, N / stride, kc, stride), lambda x, xo, y, xi, z:
        Coeff[x, 0] * B[xo, y, xi, 0, z] + Coeff[x, 1] * B[xo, y, xi, 1, z] + Coeff[x, 2] * B[xo, y, xi, 2, z] + Coeff[x, 3] * B[xo, y, xi, 3, z], 
        name="interpB")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * interpB[x, k // kc, y // nc, tvm.tir.indexmod(k, kc), tvm.tir.indexmod(y, nc)], axis=k), name='C')
    s = te.create_schedule(C.op)
    (k,) = s[C].op.reduce_axis
    (x, y) = s[C].op.axis
    ko, ki = s[C].split(k, kc)
    yo, yi = s[C].split(y, nc)
    yu, yr = s[C].split(yi, stride)
    
    s[interpB].compute_at(s[C], yr)
    s[C].reorder(yo, ko, yu, ki, x, yr)
    s[C].unroll(x)
    s[C].vectorize(yr)
    s[interpB].compute_inline()

    func = tvm.build(s, [A, B, Coeff, C], target=target, name=name)
    if print_lower:
        print(tvm.lower(s, [A, B, Coeff, C], simple_mode=True))
    return [func, packFunc, nc, kc]
