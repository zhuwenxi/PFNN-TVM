//
//  eigen_infer.hpp
//  pfnn_tvm
//
//  Created by haidonglan on 2021/7/21.
//

#ifndef eigen_infer_hpp
#define eigen_infer_hpp

#include "BaseInfer.hpp"

#include <stdio.h>
#include <string>
#include <vector>
#include <Eigen/Dense>

class EigenInfer : public BaseInfer {
public:
    EigenInfer(const int batch,
             const int input_vector_len,
             const int layer1_channel,
             const int layer2_channel,
             const int output_vector_len);
    
    EigenInfer();
    
    ~EigenInfer(){}
    
    int LoadModel(TVM_DAT *weights_layer1, TVM_DAT *bias_layer1,
                 TVM_DAT *weights_layer2, TVM_DAT *bias_layer2,
                 TVM_DAT *weights_layer3, TVM_DAT *bias_layer3);
    
    void SetInput(int vector_index, TVM_DAT value);
    void SetInput(int index_in_batch, int vector_index, TVM_DAT value);
    
    int Forward(std::vector<float> phases);
    
    TVM_DAT GetOutput(int index_in_batch, int vector_index);
    TVM_DAT GetOutput(int vector_index);
    
private:
    // Pre-allocated DLTensors, may have multiple copies
    std::vector<Eigen::ArrayXXf> w1;
    std::vector<Eigen::ArrayXXf> w2;
    std::vector<Eigen::ArrayXXf> w3;

    std::vector<Eigen::ArrayXf> b1;
    std::vector<Eigen::ArrayXf> b2;
    std::vector<Eigen::ArrayXf> b3;

    // Dynamic DLTensors
    Eigen::ArrayXXf inputTensor;
    Eigen::ArrayXXf outputTensor;

#ifdef EIGEN_INFER_BATCH_MODE
    Eigen::ArrayXXf w1_combo;
    Eigen::ArrayXXf w2_combo;
    Eigen::ArrayXXf w3_combo;
    Eigen::ArrayXf b1_combo;
    Eigen::ArrayXf b2_combo;
    Eigen::ArrayXf b3_combo;
#endif
};

#endif /* eigen_infer_hpp */
