import tvm
from tvm import te

dtype = "float32"

def gemm_cubic_interp_A_kernel_fuseK_with_libxsmm(dim, target, name='defaultGEMMCubicInterpolation', print_lower=False):
    M = dim[0]
    N = dim[1]
    K = dim[2]
    kc = K
    nc = 32
    mc = M
    W = 4 # Four copies of weights, constant.
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

    def intrin_libxsmm(m, k, n):
        a = te.placeholder((m, k), name='a')
        b = te.placeholder((k, n), name='b')
        k_axis = te.reduce_axis((0, k), name='k')
        c = te.compute((m, n), lambda i, j: te.sum(a[i, k_axis] * b[k_axis, j], axis=k_axis), name='c')
        a_buffer = tvm.tir.decl_buffer(a.shape, a.dtype, name='a_buffer', offset_factor=1, strides=[te.var('s1'), 1])#[te.var('s1'), te.var('s11')])
        b_buffer = tvm.tir.decl_buffer(b.shape, b.dtype, name='b_buffer', offset_factor=1, strides=[te.var('s2'), 1])
        c_buffer = tvm.tir.decl_buffer(c.shape, c.dtype, name='c_buffer', offset_factor=1, strides=[te.var('s3'), 1])

        def intrin_func(ins, outs):
            def _body():
                ib = tvm.tir.ir_builder.create()
                ib.emit(
                tvm.tir.call_extern(
                    "int", "call_libxsmm_sgemm", m, n, k, 1.0, ins[0].access_ptr("r"), K * 4, ins[1].access_ptr("r"), n, 0.0, outs[0].access_ptr("w"), N
                )
                )
                return ib.get()
            def _update():
                ib = tvm.tir.ir_builder.create()
                ib.emit(
                tvm.tir.call_extern(
                    "int", "call_libxsmm_sgemm", m, n, k, 1.0, ins[0].access_ptr("r"), K * 4, ins[1].access_ptr("r"), n, 1.0, outs[0].access_ptr("w"), N
                )
                )
                return ib.get()

            return _body(), None, _update(), _update(), _body()
        return te.decl_tensor_intrin(c.op, intrin_func, binds={a: a_buffer, b: b_buffer, c: c_buffer})

    s = te.create_schedule(C.op)
    (x, y) = s[C].op.axis
    (reduce_k, ) = s[C].op.reduce_axis
    xo, xi = s[C].split(x, mc)
    ko, ki = s[C].split(reduce_k, kc)
    yt, yu = s[C].split(y, nc)
    yo, yi = s[C].split(yu, stride)
    s[C].reorder(xo, ko, yt, yo, ki, xi, yi)

    # with libxsmm
    micro_kernel = intrin_libxsmm(mc, kc, nc)
    s[C].tensorize(ki, micro_kernel)

    func = tvm.build(s, [A, B, Coeff, C], target=target, name=name)
    if print_lower:
        print(tvm.lower(s, [A, B, Coeff, C], simple_mode=True))
    return [func, packFunc, nc, kc]
