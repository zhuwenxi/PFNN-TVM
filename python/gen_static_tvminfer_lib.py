"""Testcode for Android RPC GEMM.
"""

import tvm
import os
from tvm.contrib import utils
import shutil
import argparse
import subprocess

from template.manual.tvm_activation_template import activation_cubic_interp_kernel
from utils.performance_replay import schedule_performance_replay_interpB, schedule_performance_replay_interpA

# The default tensor type in tvm
dtype = "float32"

def tvm_interpB(dim, target, name='defaultGEMMCubicInterpolation', print_lower=True):
    from template.manual.tvm_interpB_template import gemm_cubic_interp_B_kernel
    func, packFunc, nc, kc = gemm_cubic_interp_B_kernel(dim, target, name, print_lower)
    if performance_replay:
        schedulePerformanceReplayInterpB(dim, target, func, packFunc, nc, kc)
    return [func, packFunc]

def tvm_interpA(dim, target, name='defaultGEMMCubicInterpolation', print_lower=True):
    from template.manual.tvm_interpA_template import gemm_cubic_interp_A_kernel_fuseK
    func, packFunc, nc, kc = gemm_cubic_interp_A_kernel_fuseK(dim, target, name, print_lower)
    if performance_replay:
        schedulePerformanceReplayInterpA(dim, target, func, packFunc, nc, kc)
    return [func, packFunc]

def tvm_interpA_with_xsmm(dim, target, name='defaultGEMMCubicInterpolation', print_lower=False):
    from template.manual.tvm_interpA_extern_xsmm_template import gemm_cubic_interp_A_kernel_fuseK_with_xsmm
    func, packFunc, nc, kc = gemm_cubic_interp_A_kernel_fuseK_with_xsmm(dim, target, name, print_lower)
    if performance_replay:
        schedulePerformanceReplayInterpA(dim, target, func, packFunc, nc, kc)
    return [func, packFunc]

def autotvm_interpB(dim, target, name='defaultGEMMCubicInterpolation', print_lower=True):
    from template.autoTVM.autotvm_interpB_template import autotune_interp_B_kernel
    func, packFunc, nc, kc = autotune_interp_B_kernel(dim, target, name, use_tune, print_lower)
    if performance_replay:
        schedulePerformanceReplayInterpB(dim, target, func, packFunc, nc, kc)
    return [func, packFunc]

def autotvm_interpA(dim, target, name='defaultGEMMCubicInterpolation', print_lower=True):
    from template.autoTVM.autotvm_interpA_template import autotune_interp_A_kernel
    func, packFunc, nc, kc = autotune_interp_A_kernel(dim, target, name, use_tune, print_lower)
    if performance_replay:
        schedulePerformanceReplayInterpA(dim, target, func, packFunc, nc, kc)
    return [func, packFunc]

def autotvm_interpA_with_xsmm(dim, target, name='defaultGEMMCubicInterpolation', print_lower=False):
    from template.autoTVM.autotvm_interpA_extern_xsmm_template import autotune_interp_A_with_xsmm_kernel
    func, packFunc, nc, kc = autotune_interp_A_with_xsmm_kernel(dim, target, name, use_tune, print_lower)
    if performance_replay:
        schedulePerformanceReplayInterpA(dim, target, func, packFunc, nc, kc)
    return [func, packFunc]

def ansor_interpB(dim, target, name='defaultGEMMCubicInterpolation', print_lower=True):
    from template.ansor.ansor_interpB_template import ansor_tune_interp_B_kernel
    func, packFunc, nc, kc = ansor_tune_interp_B_kernel(dim, target, name, use_tune, print_lower)
    if performance_replay:
        schedulePerformanceReplayInterpB(dim, target, func, packFunc, nc, kc)
    return [func, packFunc]

def ansor_interpA(dim, target, name='defaultGEMMCubicInterpolation', print_lower=True):
    from template.ansor.ansor_interpA_template import ansor_tune_interp_A_kernel
    func, packFunc, nc, kc = ansor_tune_interp_A_kernel(dim, target, name, use_tune, print_lower)
    if performance_replay:
        schedulePerformanceReplayInterpA(dim, target, func, packFunc, nc, kc)
    return [func, packFunc]


def gen_lib_complete(batch, inVecLen, hiddenVecLen, outVecLen, prefix, func=None, print_lower=True):
    K1 = inVecLen
    N1 = hiddenVecLen
    K2 = hiddenVecLen
    N2 = hiddenVecLen
    K3 = hiddenVecLen
    N3 = outVecLen
    layer_shapes = [(batch, N1, K1), (batch, N2, K2), (batch, N3, K3)]
    
    # Small GEMM with cubic interp
    mod_list = []
    
    mod = func(layer_shapes[0], target, name='{}_fc1_bs{}'.format(prefix, batch), print_lower = print_lower)
    mod_list += mod
    mod = func(layer_shapes[1], target, name='{}_fc2_bs{}'.format(prefix, batch), print_lower = print_lower)
    mod_list += mod
    mod = func(layer_shapes[2], target, name='{}_fc3_bs{}'.format(prefix, batch), print_lower = print_lower)
    mod_list += mod
    
    # Acitvations
    mod = activation_cubic_interp_kernel((batch, N2), target, name='{}_activation_bias_elu_bs{}_vec{}'.format(prefix, batch, N2), with_elu=True)
    mod_list += mod
    mod = activation_cubic_interp_kernel((batch, N3), target, name='{}_activation_bias_bs{}_vec{}'.format(prefix, batch, N3), with_elu=False)
    mod_list += mod
    return mod_list

def write_dummy_stub_file(mod_list):
    with open('dummy_stub.cc', 'w') as f:
        f.write('#define TVM_FUNC_STATIC_STUB(mod_name) \\\n')
        f.write('\tvoid mod_name(void*, void*, int, void*, void*, void*);\n')
        f.write('#define TVM_FUNC_STATIC_STUB_REFERENCE(mod_name) \\\n')
        f.write('\tmod_name(h, h, 0, h, h, h);\n')
        f.write('extern "C" {\n')
        for mod in mod_list:
            func_name = mod.name
            f.write('\tTVM_FUNC_STATIC_STUB({})\n'.format(func_name))
        f.write('}\n')
        f.write('void stub() {\n')
        f.write('\tvoid* h;\n')
        for mod in mod_list:
            func_name = mod.name
            f.write('\tTVM_FUNC_STATIC_STUB_REFERENCE({})\n'.format(func_name))
        f.write('}\n')
        f.write('#undef TVM_FUNC_STATIC_STUB\n')
        f.write('#undef TVM_FUNC_STATIC_STUB_REFERENCE\n')
        
def compile_static_lib(lib_path, mod_list, tmpdir):
    obj_file_list = []
    for mod in mod_list:
        obj_file_path = tmpdir.relpath('{}.o'.format(mod.name))
        mod.save(obj_file_path)
        obj_file_list.append(obj_file_path)
    subprocess.run(["ar", "qc", lib_path] + obj_file_list)

def export_module(target, sdk, arch, shape_configs, func):
    lib_name = "libtvminfer_{}_{}.a".format(sdk, arch)
    temp = utils.tempdir()
    path_dso = temp.relpath(lib_name)
    mod_list = []
    for prefix in shape_configs:
        shape = shape_configs[prefix]
        K1 = shape[0]
        K2 = shape[1]
        N3 = shape[2]
        for batch in (1, 5, 8):
            mod_list = mod_list + gen_lib_complete(batch, K1, K2, N3, prefix, func, print_lower=False)
    
    compile_static_lib(path_dso, mod_list, temp)
    if not use_tune:
        write_dummy_stub_file(mod_list)
        shutil.copyfile(path_dso, "./{}".format(lib_name))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TVM compute library for PFNN with cubic optimization for Android')
    parser.add_argument('-p', '--platform', dest='platform', default = 'android',
            help='select platform: android or linux')
    parser.add_argument('-d', '--hidden', dest='hidden_len', default = 256, type=int,
            help='Hidden layer size.')
    parser.add_argument('--mode', dest='mode', default='manual', choices=['manual', 'autoTVM', 'ansor'], 
            help='select scheduler mode manual, autoTVM or ansor')
    parser.add_argument('--interp', dest='interp', default='B', choices=['A', 'B'],
            help='select interpolation: A or B')
    parser.add_argument('--extern', dest='extern', action="store_true", 
            help='whether to use extern micro kernel')
    parser.add_argument('--tune', dest='use_tune', action="store_true", 
            help='whether to use tune mode')
    args = parser.parse_args()

    if args.platform == "android":
        ll = "llvm --system-lib"
        arch = "aarch64"
        sdk = "android"
        target = "%s -mtriple=%s-linux-android" % (ll, arch)
        import config.android_config
    elif args.platform == "linux":
        if args.extern:
            raise Exception("error: linux using extern micro kernel requires an additional environment (this version not supported yet)!")
        ll = "llvm --system-lib"
        arch = "x86_64"
        sdk = "linux"
        target = "%s -mcpu=skylake-avx512" % (ll)
        # target = "%s -mcpu=core-avx2" % (ll)
        import config.linux_config
    else:
        raise Exception("error: Unknow platform {}".format(args.platform))

    K2 = args.hidden_len

    use_tune = args.use_tune
    performance_replay = False

    template_name = args.mode + '_' + args.interp
    if args.extern : template_name += '_extern'
    print(template_name)

    shape_configs = {"male":(1032, K2, 912)}
    func_dict = {
        "manual_B": tvm_interpB,
        "manual_A": tvm_interpA,
        "manual_A_extern": tvm_interpA_with_xsmm,
        "autoTVM_B": autotvm_interpB,
        "autoTVM_A": autotvm_interpA,
        "autoTVM_A_extern": autotvm_interpA_with_xsmm,
        "ansor_B": ansor_interpB,
        "ansor_A": ansor_interpA,
    }
    if template_name not in func_dict:
        raise Exception("error: interB can not use_extern or interpA with ansor can not use_extern")
        
    export_module(target, sdk, arch, shape_configs, func_dict.get(template_name))
