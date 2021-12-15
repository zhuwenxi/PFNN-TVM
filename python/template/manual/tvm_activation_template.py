import tvm
from tvm import te

dtype = "float32"

def activation_cubic_interp_kernel(dim, target, name="bias_cubic_interp", with_elu=True):
    M = dim[0]
    N = dim[1]
    # Specific parameters
    stride = 16
    W = 4
    Coeff = te.placeholder((M, 4), dtype=dtype, name='coeffcient')

    #Pack func
    BI = te.placeholder((W, N), name="Bias")
    packBI = te.compute((N // stride, W, stride), lambda x, w, z: BI[w, x * stride + z], name="packBias")
    s_packb = te.create_schedule(packBI.op)
    #print(tvm.lower(s_packb, [BI, packBI], simple_mode=False))
    packbFunc = tvm.build(s_packb, [BI, packBI], target=target, name="{}_pack".format(name))

    O = te.placeholder((M, N), dtype=dtype)
    Bias = te.placeholder((N // stride, W, stride), dtype=dtype)
    InterpBias = te.compute((M, N // stride, stride), lambda x, y, z :
        Coeff[x, 0] * Bias[y, 0, z] + Coeff[x, 1] * Bias[y, 1, z] + Coeff[x, 2] * Bias[y, 2, z] + Coeff[x, 3] * Bias[y, 3, z])
    # BiasAdd = te.compute((M, N), lambda x, y: O[x, y] + InterpBias[x, y // stride, tvm.tir.indexmod(y, stride)])
    BiasAdd = te.compute((M, N), lambda x, y: O[x, y] + InterpBias[x, y // stride, tvm.tir.indexmod(y, stride)])
    s_bias = te.create_schedule(BiasAdd.op)

    if with_elu:
        exp = te.compute((M, N), lambda x, y : te.exp(te.min(BiasAdd[x, y], 0)))
        E = te.compute((M, N), lambda x, y : te.max(0, BiasAdd[x, y]) + exp[x, y] - 1, name='ELU')
        s_bias = te.create_schedule(E.op)

    x, y = BiasAdd.op.axis
    s_bias[InterpBias].compute_at(s_bias[BiasAdd], y)
    s_bias[BiasAdd].reorder(y, x)
    s_bias[BiasAdd].unroll(x)
    yo, yi = s_bias[BiasAdd].split(y, stride)
    s_bias[BiasAdd].vectorize(yi)

    if with_elu:
        s_bias[exp].compute_inline()
        s_bias[BiasAdd].compute_at(s_bias[E], E.op.axis[1])
        
        #print(tvm.lower(s_bias, [O, Bias, Coeff, E], simple_mode=False))
        biasFunc = tvm.build(s_bias, [O, Bias, Coeff, E], target=target, name="{}".format(name))
    else:
        #print(tvm.lower(s_bias, [O, Bias, Coeff, BiasAdd], simple_mode=False))
        biasFunc = tvm.build(s_bias, [O, Bias, Coeff, BiasAdd], target=target, name="{}".format(name))

    return [biasFunc, packbFunc]