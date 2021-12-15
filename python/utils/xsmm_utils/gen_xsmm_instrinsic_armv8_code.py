def xsmm_instrinsic_armv8_code(M, K, N, lda, ldb, ldc, uniq_id):
    def compile_time_for_compute_unroll_k(M, N, K, lda, ldb, ldc, COLS, LINES, unroll_nums):
        assert(unroll_nums % 4 == 0)
        code_str = ""
        code_str += f"    int k = 0;\n"
        code_str += f"    for (; k + {unroll_nums} <= {K}; k += {unroll_nums}) {{\n"
        for k_unroll in range(0, unroll_nums, 4):
          for i in range(LINES):
            code_str += f"      va[{i}] = vld1q_f32(A + k + {k_unroll} + m * {lda} + {i * lda});\n"
          for unrool_4 in range(4):
            for col in range(COLS):
              code_str += f"      vb[{col}] = vld1q_f32(B + k * {ldb} + {k_unroll * ldb +  unrool_4 * ldb + col * 4});\n"
            for i in range(LINES*COLS):
                line = i // COLS
                col = i % COLS
                code_str += f"      vc[{line*COLS+col}] = vmlaq_laneq_f32(vc[{line*COLS+col}], vb[{col}], va[{line}], {unrool_4});\n"
        code_str += f"    }}\n"
        if (K % unroll_nums):
          remain = K % unroll_nums
          for k_unroll in range(0, remain, 4):
            unroll_num = 4
            if(remain % 4 != 0 and k_unroll + 4  >= remain):
              unroll_num = remain % 4
            for i in range(LINES):
              code_str += f"    va[{i}] = vld1q_f32(A + k + {k_unroll} + m * {lda} + {i * lda});\n"
            for unrool_4 in range(unroll_num):
              for col in range(COLS):
                code_str += f"    vb[{col}] = vld1q_f32(B + k * {ldb} + {k_unroll * ldb +  unrool_4 * ldb + col * 4});\n"
              for i in range(LINES*COLS):
                  line = i // COLS
                  col = i % COLS
                  code_str += f"    vc[{line*COLS+col}] = vmlaq_laneq_f32(vc[{line*COLS+col}], vb[{col}], va[{line}], {unrool_4});\n"
          
        return code_str

    def compile_time_for_set0(M, N, K, lda, ldb, ldc, COLS, LINES):
        code_str = ""
        for i in range(LINES*COLS):
            code_str += f"    vc[{i}] = vdupq_n_f32((float32_t)0);\n"
        return code_str

    def compile_time_for_load(M, N, K, lda, ldb, ldc, COLS, LINES):
        code_str = ""
        for i in range(LINES*COLS):
            line = i // COLS
            col = i % COLS
            code_str += f"    vc[{i}] = vld1q_f32(C + m * {ldc} + {line * ldc + col * 4});\n"
        return code_str

    def compile_time_for_store(M, N, K, lda, ldb, ldc, COLS, LINES):
        MASK = N % 4
        code_str = ""
        for i in range(LINES*COLS):
            line = i // COLS
            col = i % COLS
            if (col == COLS-1):
              if MASK == 0:
                code_str += f"    vst1q_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}]);\n"
              if MASK == 1:
                code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}], 0);\n"
              if MASK == 2:
                code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}], 0);\n"
                code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4 + 1}, vc[{i}], 1);\n"
              if MASK == 3:
                code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}], 0);\n"
                code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4 + 1}, vc[{i}], 1);\n"
                code_str += f"    vst1q_lane_f32(C + m * {ldc} + {line * ldc + col * 4 + 2}, vc[{i}], 2);\n"
            else:
              code_str += f"    vst1q_f32(C + m * {ldc} + {line * ldc + col * 4}, vc[{i}]);\n"
        return code_str

    
    NC = min(N, 32)
    UNROLL_NUM = 4
    EXPANDED_NC = ((NC-1)//4 + 1) * 4
    COLS = EXPANDED_NC // 4
    LINES = (32 - COLS) // (COLS + 1) 
    UNROLL_NUM = 4
    """Emit C code for gemm impl."""
    cc_code = f"""
#ifndef __SGEMM_KERNEL_H
#define __SGEMM_KERNEL_H
#endif
#include <cmath>
#include <cstring>
#include <cassert>
#include <arm_neon.h>
#include <cstdlib>
#include <cstdio>

namespace laf {{
void small_gemm_fixmn(const float *A, const float *B, float *C) {{

  float32x4_t va[{LINES}];
  float32x4_t vb[{COLS}];
  float32x4_t vc[{LINES * COLS}];

  int m = 0;
  for (; m + {LINES} <= {M}; m += {LINES}) {{
"""
    cc_code += compile_time_for_set0(M, NC, K, lda, ldb, ldc, COLS, LINES)
    cc_code += compile_time_for_compute_unroll_k(M, NC, K, lda, ldb, ldc, COLS, LINES, UNROLL_NUM)
    cc_code += compile_time_for_store(M, NC, K, lda, ldb, ldc, COLS, LINES)
    cc_code += f"""
  }}
"""
    if M % LINES:
      lines = M % LINES
      cc_code += compile_time_for_set0(M, NC, K, lda, ldb, ldc, COLS, lines)
      cc_code += compile_time_for_compute_unroll_k(M, NC, K, lda, ldb, ldc, COLS, lines,  UNROLL_NUM)
      cc_code += compile_time_for_store(M, NC, K, lda, ldb, ldc, COLS, lines)
    cc_code += f"""
}}
void small_gemm_fixmn_with_bias(const float *A, const float *B, float *C) {{

  float32x4_t va[{LINES}];
  float32x4_t vb[{COLS}];
  float32x4_t vc[{LINES * COLS}];

  int m = 0;
  for (; m + {LINES} <= {M}; m += {LINES}) {{
"""
    cc_code += compile_time_for_load(M, NC, K, lda, ldb, ldc, COLS, LINES)
    cc_code += compile_time_for_compute_unroll_k(M, NC, K, lda, ldb, ldc, COLS, LINES, UNROLL_NUM)
    cc_code += compile_time_for_store(M, NC, K, lda, ldb, ldc, COLS, LINES)
    cc_code += f"""
  }}
"""
    if M % LINES:
      lines = M % LINES
      cc_code += compile_time_for_load(M, NC, K, lda, ldb, ldc, COLS, lines)
      cc_code += compile_time_for_compute_unroll_k(M, NC, K, lda, ldb, ldc, COLS, lines,  UNROLL_NUM)
      cc_code += compile_time_for_store(M, NC, K, lda, ldb, ldc, COLS, lines)
    cc_code += f"""
}}
"""
    REMAIN_N = N % NC
    if REMAIN_N:
      EXPANDED_N = ((REMAIN_N-1)//4 + 1) * 4
      REMAIN_COLS = EXPANDED_N // 4
      REMAIN_LINES = (32 - REMAIN_COLS) // (REMAIN_COLS + 1) 
      cc_code += f"""
void small_gemm_fixn(const float *A, const float *B, float *C) {{

  float32x4_t va[{REMAIN_LINES}];
  float32x4_t vb[{REMAIN_COLS}];
  float32x4_t vc[{REMAIN_LINES * REMAIN_COLS}];

  int m = 0;
  for (; m + {REMAIN_LINES} <= {M}; m += {REMAIN_LINES}) {{
"""
      cc_code += compile_time_for_set0(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, REMAIN_LINES)
      cc_code += compile_time_for_compute_unroll_k(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, REMAIN_LINES, UNROLL_NUM)
      cc_code += compile_time_for_store(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, REMAIN_LINES)
      cc_code += f"""
  }}
"""
      if M % REMAIN_LINES:
        remain_lines = M % REMAIN_LINES
        cc_code += compile_time_for_set0(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, remain_lines)
        cc_code += compile_time_for_compute_unroll_k(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, remain_lines,  UNROLL_NUM)
        cc_code += compile_time_for_store(M, REMAIN_N, K, lda, ldb, ldc, COLS, remain_lines)
      cc_code += f"""
}}
void small_gemm_fixn_with_bias(const float *A, const float *B, float *C) {{

  float32x4_t va[{REMAIN_LINES}];
  float32x4_t vb[{REMAIN_COLS}];
  float32x4_t vc[{REMAIN_LINES * REMAIN_COLS}];

  int m = 0;
  for (; m + {REMAIN_LINES} <= {M}; m += {REMAIN_LINES}) {{
"""
      cc_code += compile_time_for_load(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, REMAIN_LINES)
      cc_code += compile_time_for_compute_unroll_k(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, REMAIN_LINES, UNROLL_NUM)
      cc_code += compile_time_for_store(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, REMAIN_LINES)
      cc_code += f"""
  }}
"""
      if M % REMAIN_LINES:
        lines = M % REMAIN_LINES
        cc_code += compile_time_for_load(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, remain_lines)
        cc_code += compile_time_for_compute_unroll_k(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, remain_lines,  UNROLL_NUM)
        cc_code += compile_time_for_store(M, REMAIN_N, K, lda, ldb, ldc, REMAIN_COLS, remain_lines)
      cc_code += f"""
}}
"""
    cc_code += f"""
}}

extern "C" int gemm_{M}x{K}x{N}_{lda}_{ldb}_{ldc}_xsmm_{uniq_id}(
    const float *A, const float *B, float *C) {{
    int i=0;
    for (; i+{NC}<={N}; i+={NC})
      laf::small_gemm_fixmn(A, B + i, C + i);
  """
    if REMAIN_N:
      cc_code += f"""laf::small_gemm_fixn(A, B + i, C + i);"""
    cc_code += f"""
    return 0;
}}
extern "C" int gemm_{M}x{K}x{N}_{lda}_{ldb}_{ldc}_xsmm_with_bias_{uniq_id}(
    const float *A, const float *B, float *C) {{
    int i=0;
    for (; i+{NC}<={N}; i+={NC})
      laf::small_gemm_fixmn_with_bias(A, B + i, C + i);
    """
    if REMAIN_N:
      cc_code += f"""laf::small_gemm_fixn_with_bias(A, B + i, C + i);"""
    cc_code += f"""
        return 0;
}}
    """
    return cc_code