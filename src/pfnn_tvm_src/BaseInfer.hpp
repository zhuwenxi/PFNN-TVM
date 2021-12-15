//
//  BaseInfer.hpp
//  pfnn_tvm
//
//  Created by haidonglan on 2021/7/27.
//

#ifndef BaseInfer_hpp
#define BaseInfer_hpp

#include <stdio.h>
#include <vector>

typedef float TVM_DAT;

class BaseInfer {
public:
    BaseInfer(const int batch,
             const int input_vector_len,
             const int layer1_channel,
             const int layer2_channel,
             const int output_vector_len);
    
    virtual ~BaseInfer();
    
    void LoadWeightsFromFiles(const char* dirpath);
    
    virtual int LoadModel(TVM_DAT *weights_layer1, TVM_DAT *bias_layer1,
                         TVM_DAT *weights_layer2, TVM_DAT *bias_layer2,
                         TVM_DAT *weights_layer3, TVM_DAT *bias_layer3) {
        return 0;
    }
    
    virtual int AddNormalizeVector(TVM_DAT* Xmean_, TVM_DAT* Xstd_, TVM_DAT* Ymean_, TVM_DAT* Ystd_) {
        return -914;
    }
    
    virtual void SetInput(int vector_index, TVM_DAT value) {}
    virtual void SetInput(int index_in_batch, int vector_index, TVM_DAT value) {}
    
    virtual int Forward(std::vector<float> model_index) {
        return -914;
    }
    
    virtual TVM_DAT GetOutput(int index_in_batch, int vector_index){return 0.f;}

    TVM_DAT GetYmean(int vector_index);
    
protected:
    int batch;
    // Static Shapes
    int K1;
    int N1;
    int K2;
    int N2;
    int K3;
    int N3;
    int W;
    
    float *Xmean;
    float *Xstd;
    float *Ymean;
    float *Ystd;
private:
    int load_weights(TVM_DAT* buffer, int rows, int cols, FILE* f);
    int load_weights(TVM_DAT* buffer, int rows, int cols, const char* filename);
    int load_weights(TVM_DAT* buffer, int weights_num, int rows, int cols, const char* filename);
};

#endif /* BaseInfer_hpp */
