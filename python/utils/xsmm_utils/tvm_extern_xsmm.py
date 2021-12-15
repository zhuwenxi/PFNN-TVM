import os
import tvm
from tvm import te
from tvm.contrib import utils, clang
import random
import string
from utils.xsmm_utils.gen_xsmm_instrinsic_armv8_code import xsmm_instrinsic_armv8_code
from config.android_config import cc_compiler

def intrin_gemm_MxKxN(M, K, N, m, k, n):
    """Defines a SIMD-accelerated transposed matmul."""
    # we generate a unique ID for every intrinsic definition, to prevent name
    # collisions in the generated source (e.g., if there are multiple operators
    # in the same module that use the same intrinsic)
    #
    # TODO(weberlo, areusch): to cut down on memory usage, we should cache each intrinsic
    # instantiation and include it only once, eliminating the need for unique
    # IDs
    UNIQ_ID_LEN = 8
    uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))

    a = te.placeholder((m, k), name='a')
    b = te.placeholder((k, n), name='b')
    k_axis = te.reduce_axis((0, k), name='k')
    c = te.compute((m, n), lambda i, j: te.sum(a[i, k_axis] * b[k_axis, j], axis=k_axis), name='c')
    a_buffer = tvm.tir.decl_buffer(a.shape, a.dtype, name='a_buffer', offset_factor=1, strides=[te.var('s1'), 1])
    b_buffer = tvm.tir.decl_buffer(b.shape, b.dtype, name='b_buffer', offset_factor=1, strides=[te.var('s2'), 1])
    c_buffer = tvm.tir.decl_buffer(c.shape, c.dtype, name='c_buffer', offset_factor=1, strides=[te.var('s3'), 1])
    def intrin_func(ins, outs):
        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                    tvm.tir.call_extern(
                        "int32",
                        f"gemm_{m}x{k}x{n}_{K}_{n}_{N}_xsmm_{uniq_id}",
                        ins[0].access_ptr("r"), 
                        ins[1].access_ptr("r"), 
                        outs[0].access_ptr("w")
                    )
            )
            return ib.get()
        def _update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                    tvm.tir.call_extern(
                        "int32",
                        f"gemm_{m}x{k}x{n}_{K}_{n}_{N}_xsmm_with_bias_{uniq_id}",
                        ins[0].access_ptr("r"), 
                        ins[1].access_ptr("r"), 
                        outs[0].access_ptr("w")
                    )
            )
            return ib.get()
        return _body(), None, _update()

    intrin_decl = te.decl_tensor_intrin(c.op, intrin_func, binds={a: a_buffer, b: b_buffer, c: c_buffer})
    return intrin_decl, uniq_id


def gemm_MxKxN_impl(M, K, N, lda, ldb, ldc, uniq_id):
    # Create c source code
    cc_code = xsmm_instrinsic_armv8_code(M, K, N, lda, ldb, ldc, uniq_id)

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=["-march=armv8-a", "-O3", "-std=c++14", "-static"], cc=cc_compiler)
    return ll_code