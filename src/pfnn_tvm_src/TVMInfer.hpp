//
//  forward.hpp
//  tvm_runtime_packed (iOS)
//
//  Created by haidonglan on 2021/7/20.
//

#ifndef tvm_infer_hpp
#define tvm_infer_hpp

#include "BaseInfer.hpp"
#include <string>
#include <stdio.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>

typedef std::vector<int> vec;
typedef std::unordered_map<std::string, vec> tvmParamMap;

class TVMInfer : public BaseInfer {
public:
    TVMInfer(std::string libPath, const int batch,
             const int input_vector_len,
             const int layer1_channel,
             const int layer2_channel,
             const int output_vector_len);
    
    ~TVMInfer();
    
    int LoadModel(TVM_DAT *weights_layer1, TVM_DAT *bias_layer1,
                 TVM_DAT *weights_layer2, TVM_DAT *bias_layer2,
                 TVM_DAT *weights_layer3, TVM_DAT *bias_layer3);
    
    int AddNormalizeVector(TVM_DAT* Xmean, TVM_DAT* Xstd, TVM_DAT* Ymean, TVM_DAT* Ystd);
    
    void SetInput(int vector_index, TVM_DAT value);
    void SetInput(int index_in_batch, int vector_index, TVM_DAT value);
    
    int Forward(std::vector<float> phase);
    
    TVM_DAT GetOutput(int index_in_batch, int vector_index);
    TVM_DAT GetOutput(int vector_index);

private:
    tvmParamMap tvm_param_table;

    void MakeCoeff(std::vector<float>& phases, float* coeff_buffer);
    int ext_N3;
    tvm::runtime::PackedFunc func_fc1;
    tvm::runtime::PackedFunc func_fc1_pack;
    tvm::runtime::PackedFunc func_fc2;
    tvm::runtime::PackedFunc func_fc2_pack;
    tvm::runtime::PackedFunc func_fc3;
    tvm::runtime::PackedFunc func_fc3_pack;
    tvm::runtime::PackedFunc func_b1;
    tvm::runtime::PackedFunc func_b1_pack;
    tvm::runtime::PackedFunc func_b2;
    tvm::runtime::PackedFunc func_b2_pack;
    tvm::runtime::PackedFunc func_b3;
    tvm::runtime::PackedFunc func_b3_pack;
    
    // Pre-allocated DLTensors, may have multiple copies
    DLTensor* packed_w1;
    DLTensor* packed_w2;
    DLTensor* packed_w3;
    DLTensor* packed_b1;
    DLTensor* packed_b2;
    DLTensor* packed_b3;
    DLTensor* coeff;
    // std::vector<DLTensor*> w1;
    // std::vector<DLTensor*> w2;
    // std::vector<DLTensor*> w3;
    
    // std::vector<DLTensor*> b1;
    // std::vector<DLTensor*> b2;
    // std::vector<DLTensor*> b3;
    
    // Dynamic DLTensors
    DLTensor* inputTensor;
    DLTensor* outputTensor;
    DLTensor* o1;
    DLTensor* o2;
    DLTensor* o3;
    DLTensor* elu1;
    DLTensor* elu2;
};
#endif /* forward_hpp */
