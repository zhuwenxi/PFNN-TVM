//
//  forward.cpp
//  tvm_runtime_packed (iOS)
//
//  Created by haidonglan on 2021/7/20.
//

#include "TVMInfer.hpp"
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include "timer.h"
#include <cmath>
#include <Eigen/Dense>
#include "json.hpp"
#include <iostream>
#include <fstream>

TVMInfer::TVMInfer(std::string libPath, const int batch,
                   const int input_vector_len,
                   const int layer1_channel,
                   const int layer2_channel,
                   const int output_vector_len) : BaseInfer(batch, input_vector_len, layer1_channel, layer2_channel, output_vector_len), func_fc1(),func_fc2(),
func_fc3(),packed_w1(NULL), packed_w2(NULL), packed_w3(NULL),packed_b1(NULL),packed_b2(NULL),packed_b3(NULL),inputTensor(NULL), outputTensor(NULL), o1(NULL), o2(NULL)
{
    using json = nlohmann::json;
    std::ifstream i("cache_block_size.json");
    json j;
    i >> j;
    for(auto it = j.begin(); it != j.end(); ++it){
        vec value = it.value();
        tvm_param_table[it.key()] = value;
    }

    ext_N3 = N3;
    // Some optional magic numbers defined for TVM generated library.
    if (N3 == 908) {
        ext_N3 = 912;
    }

    // Prefix deduction by shape
    std::string prefix;
    if (input_vector_len == 1032 && layer1_channel == 256 && layer2_channel == 256 && output_vector_len == 908) {
        prefix = "male";
    } 
    
#ifdef PFNN_USE_SYSTEM_LIB
    printf("Loading module from system library\n");
    tvm::runtime::Module mod_tvmlib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
    printf("Done\n");
#else
    printf("Loading module in shape constructor @ %s...\n", libPath.c_str());
    tvm::runtime::Module mod_tvmlib = tvm::runtime::Module::LoadFromFile(libPath);
#endif
    
    char func_name[64];
    sprintf(func_name, "%s_fc1_bs%d", prefix.c_str(), batch);
    this->func_fc1 = mod_tvmlib.GetFunction(func_name);
    sprintf(func_name, "%s_fc2_bs%d", prefix.c_str(), batch);
    this->func_fc2 = mod_tvmlib.GetFunction(func_name);
    sprintf(func_name, "%s_fc3_bs%d", prefix.c_str(), batch);
    this->func_fc3 = mod_tvmlib.GetFunction(func_name);
    // Pack functions for fc layers
    sprintf(func_name, "%s_fc1_bs%d_pack", prefix.c_str(), batch);
    this->func_fc1_pack = mod_tvmlib.GetFunction(func_name);
    sprintf(func_name, "%s_fc2_bs%d_pack", prefix.c_str(), batch);
    this->func_fc2_pack = mod_tvmlib.GetFunction(func_name);
    sprintf(func_name, "%s_fc3_bs%d_pack", prefix.c_str(), batch);
    this->func_fc3_pack = mod_tvmlib.GetFunction(func_name);

    sprintf(func_name, "%s_activation_bias_elu_bs%d_vec%d", prefix.c_str(), batch, N2);
    this->func_b1 = mod_tvmlib.GetFunction(func_name);
    sprintf(func_name, "%s_activation_bias_elu_bs%d_vec%d_pack", prefix.c_str(), batch, N2);
    this->func_b1_pack = mod_tvmlib.GetFunction(func_name);
    this->func_b2 = this->func_b1;
    this->func_b2_pack = this->func_b1_pack;
    sprintf(func_name, "%s_activation_bias_bs%d_vec%d", prefix.c_str(), batch, ext_N3);
    this->func_b3 = mod_tvmlib.GetFunction(func_name);
    sprintf(func_name, "%s_activation_bias_bs%d_vec%d_pack", prefix.c_str(), batch, ext_N3);
    this->func_b3_pack = mod_tvmlib.GetFunction(func_name);

    ICHECK(this->func_fc1 != nullptr);
    ICHECK(this->func_fc1_pack != nullptr);
    ICHECK(this->func_fc2 != nullptr);
    ICHECK(this->func_fc2_pack != nullptr);
    ICHECK(this->func_fc3 != nullptr);
    ICHECK(this->func_fc3_pack != nullptr);
    ICHECK(this->func_b1 != nullptr);
    ICHECK(this->func_b1_pack != nullptr);
    ICHECK(this->func_b2 != nullptr);
    ICHECK(this->func_b2_pack != nullptr);
    ICHECK(this->func_b3 != nullptr);
    ICHECK(this->func_b3_pack != nullptr);
    printf("Function load OK.\n");

    // Allocate consts
    const int dtype_code = kDLFloat;
    const int dtype_bits = 32;
    const int dtype_lanes = 1;
    const int device_type = kDLCPU;
    const int device_id = 0;
    int ndim = 2;
    
    // Input and Output tensors
    int64_t shape_i1[2] = {this->batch, K1};
    int64_t shape_o1[2] = {this->batch, N1};
    int64_t shape_o2[2] = {this->batch, N2};
    int64_t shape_o3[2] = {this->batch, ext_N3};
    int64_t shape_elu1[2] = {this->batch, N1};
    int64_t shape_elu2[2] = {this->batch, N2};
    int64_t shape_coeff[2] = {this->batch, this->W};
    
    TVMArrayAlloc(shape_i1, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &inputTensor);
    TVMArrayAlloc(shape_o1, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &o1);
    TVMArrayAlloc(shape_o2, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &o2);
    TVMArrayAlloc(shape_o3, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &o3);
    TVMArrayAlloc(shape_o3, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &outputTensor);
    TVMArrayAlloc(shape_elu1, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &elu1);
    TVMArrayAlloc(shape_elu2, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &elu2);

    TVMArrayAlloc(shape_coeff, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &coeff);
    
}

TVMInfer::~TVMInfer() {
    if(packed_w1 != NULL) {
        TVMArrayFree(packed_w1);
    }
    if(packed_w2 != NULL) {
        TVMArrayFree(packed_w2);
    }
    if(packed_w3 != NULL) {
        TVMArrayFree(packed_w3);
    }
    if(packed_b1 != NULL) {
        TVMArrayFree(packed_b1);
    }
    if(packed_b2 != NULL) {
        TVMArrayFree(packed_b2);
    }
    if(packed_b3 != NULL) {
        TVMArrayFree(packed_b3);
    }
    if(inputTensor != NULL) {
        TVMArrayFree(inputTensor);
    }
    if(outputTensor != NULL) {
        TVMArrayFree(outputTensor);
    }
    if(o1 != NULL) {
        TVMArrayFree(o1);
    }
    if(o2 != NULL) {
        TVMArrayFree(o2);
    }
    if(o3 != NULL) {
        TVMArrayFree(o3);
    }
    if(elu1 != NULL) {
        TVMArrayFree(elu1);
    }
    if(elu2 != NULL) {
        TVMArrayFree(elu2);
    }
}

int TVMInfer::AddNormalizeVector(TVM_DAT* Xmean, TVM_DAT* Xstd, TVM_DAT* Ymean, TVM_DAT* Ystd){
    return 0;
}

int TVMInfer::LoadModel(TVM_DAT *weights_layer1, TVM_DAT *bias_layer1, TVM_DAT *weights_layer2, TVM_DAT *bias_layer2, TVM_DAT *weights_layer3, TVM_DAT *bias_layer3) {
    const int dtype_code = kDLFloat;
    const int dtype_bits = 32;
    const int dtype_lanes = 1;
    const int device_type = kDLCPU;
    const int device_id = 0;
    // For cubic interpolation
    const int W = 4;
    const int stride = 16;

    char cacheBlockName[64];
    sprintf(cacheBlockName, "M%dN%dK%d", batch, N1, K1);

    const int stride1 = tvm_param_table[cacheBlockName][0];
    const int kc1 = tvm_param_table[cacheBlockName][1];
    printf("fc1 cache Block\n");

    sprintf(cacheBlockName, "M%dN%dK%d", batch, N2, K2);

    const int stride2 = tvm_param_table[cacheBlockName][0];
    const int kc2 = tvm_param_table[cacheBlockName][1];
    printf("fc2 cache Block\n");

    sprintf(cacheBlockName, "M%dN%dK%d", batch, ext_N3, K3);

    const int stride3 = tvm_param_table[cacheBlockName][0];
    const int kc3 = tvm_param_table[cacheBlockName][1];
    printf("fc3 cache Block\n");


    // Weights
    int64_t shape_w1[3] = {W, K1, N1};
    int64_t shape_w2[3] = {W, K2, N2};
    int64_t shape_w3[3] = {W, K3, ext_N3};

#ifdef TVM_INFER_BATCH_MODE
    int64_t shape_w1_pack[3] = {N1 / stride1, K1 * W, stride1};
    int64_t shape_w2_pack[3] = {N2 / stride2, K2 * W, stride2};
    int64_t shape_w3_pack[3] = {ext_N3 / stride3, K3 * W, stride3};
#else    
    int64_t shape_w1_pack[5] = {K1 / kc1, N1 / stride1, kc1, W, stride1};
    int64_t shape_w2_pack[5] = {K2 / kc2, N2 / stride2, kc2, W, stride2};
    int64_t shape_w3_pack[5] = {K3 / kc3, ext_N3 / stride3, kc3, W, stride3};
#endif

    DLTensor* w1t(NULL);
    DLTensor* w2t(NULL);
    DLTensor* w3t(NULL);
    
    TVMArrayAlloc(shape_w1, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &w1t);
    TVMArrayAlloc(shape_w2, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &w2t);
    TVMArrayAlloc(shape_w3, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &w3t);

#ifdef TVM_INFER_BATCH_MODE
    TVMArrayAlloc(shape_w1_pack, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_w1);
    TVMArrayAlloc(shape_w2_pack, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_w2);
    TVMArrayAlloc(shape_w3_pack, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_w3);
#else
    TVMArrayAlloc(shape_w1_pack, 5, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_w1);
    TVMArrayAlloc(shape_w2_pack, 5, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_w2);
    TVMArrayAlloc(shape_w3_pack, 5, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_w3);
#endif
    
    // Copy and Packing 
    memcpy((TVM_DAT *)w1t->data, weights_layer1, sizeof(TVM_DAT) * K1 * N1 * W);
    memcpy((TVM_DAT *)w2t->data, weights_layer2, sizeof(TVM_DAT) * K2 * N2 * W);
    // Possibly different stride for layer3.
    for (int i = 0; i < W * K3; ++i) {
        memcpy(((TVM_DAT*) w3t->data) + i * ext_N3, weights_layer3 + i * N3, sizeof(TVM_DAT) * N3);
    }
    printf("Packing weights...\n");
    try {
    this->func_fc1_pack(w1t, packed_w1);
    } catch (const std::exception e){
        std::cerr << e.what() <<std::endl;
    }
    printf("Packing weights 2...\n");
    this->func_fc2_pack(w2t, packed_w2);
    printf("Packing weights 3...\n");
    this->func_fc3_pack(w3t, packed_w3);
    TVMArrayFree(w1t);
    TVMArrayFree(w2t);
    TVMArrayFree(w3t);
    printf("Done.\n");

    // Bias
    DLTensor* b1t(NULL);
    DLTensor* b2t(NULL);
    DLTensor* b3t(NULL);
    int64_t shape_b1[2] = {W, N1};
    int64_t shape_b2[2] = {W, N2};
    int64_t shape_b3[2] = {W, ext_N3};
    int64_t shape_b1_pack[3] = {N1 / stride, W, stride};
    int64_t shape_b2_pack[3] = {N2 / stride, W, stride};
    int64_t shape_b3_pack[3] = {ext_N3 / stride, W, stride};
    TVMArrayAlloc(shape_b1, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &b1t);
    TVMArrayAlloc(shape_b2, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &b2t);
    TVMArrayAlloc(shape_b3, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &b3t);
    TVMArrayAlloc(shape_b1_pack, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_b1);
    TVMArrayAlloc(shape_b2_pack, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_b2);
    TVMArrayAlloc(shape_b3_pack, 3, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &packed_b3);

    printf("Packing bias...\n");

    memcpy(b1t->data, bias_layer1, sizeof(TVM_DAT) * W * N1);
    memcpy(b2t->data, bias_layer2, sizeof(TVM_DAT) * W * N2);
    // Possibly different stride for layer3.
    for (int i = 0; i < W; ++i)
        memcpy((float*)b3t->data + i * ext_N3, bias_layer3 + i * N3, sizeof(TVM_DAT) * N3);
    this->func_b1_pack(b1t, packed_b1);
    this->func_b2_pack(b2t, packed_b2);
    this->func_b3_pack(b3t, packed_b3);
    TVMArrayFree(b1t);
    TVMArrayFree(b2t);
    TVMArrayFree(b3t);
    printf("Done.\n");
    return 0;
}

void TVMInfer::SetInput(int vector_index, TVM_DAT value) {
    TVM_DAT* ptr = (TVM_DAT*) this->inputTensor->data;
    ptr[vector_index] = value;
}

void TVMInfer::SetInput(int index_in_batch, int vector_index, TVM_DAT value) {
    TVM_DAT* row_ptr = (TVM_DAT*) this->inputTensor->data + index_in_batch * this->K1;
    row_ptr[vector_index] = value;
}

TVM_DAT TVMInfer::GetOutput(int index_in_batch, int vector_index) {
    TVM_DAT* row_ptr = ((TVM_DAT*) this->outputTensor->data) + index_in_batch * ext_N3;
    return row_ptr[vector_index];
}

void TVMInfer::MakeCoeff(std::vector<float>& phases, float* coeff_buffer) {
    for (int i = 0; i < phases.size(); ++i) {
        int pindex[4];
        float P = phases[i];
        float mu = fmod((P / (2*M_PI)) * 4, 1.0);
        pindex[1] = (int)((P / (2*M_PI)) * 4);
        pindex[0] = ((pindex[1]+3) % 4);
        pindex[2] = ((pindex[1]+1) % 4);
        pindex[3] = ((pindex[1]+2) % 4);
        int coeff_order[4];
        for (int j = 0; j < 4; ++j) {
            coeff_order[pindex[j]] = j;
        }
        float coeff_line[4];
        coeff_line[0] = -0.5 * mu * mu * mu + mu * mu - 0.5 * mu;
        coeff_line[1] = 1.5 * mu * mu * mu - 2.5 * mu * mu + 1;
        coeff_line[2] = -1.5 * mu * mu * mu + 2 * mu * mu + 0.5 * mu;
        coeff_line[3] = 0.5 * mu * mu * mu - 0.5 * mu * mu;
        for (int j = 0; j < 4; ++j) {
            coeff_buffer[i * 4 + j] = coeff_line[coeff_order[j]];
        }
    }
}

int TVMInfer::Forward(std::vector<float> phases) {    
    MakeCoeff(phases, (TVM_DAT *) coeff->data);
    Timer computeTime;
    
    this->func_fc1(inputTensor, packed_w1, coeff, o1);
    this->func_b1(o1, packed_b1, coeff, elu1);
    this->func_fc2(elu1, packed_w2, coeff, o2);
    this->func_b2(o2, packed_b2, coeff, elu2);
    this->func_fc3(elu2, packed_w3, coeff, o3);
    this->func_b3(o3, packed_b3, coeff, outputTensor);
    return 0;
}

