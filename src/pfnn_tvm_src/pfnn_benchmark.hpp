//
//  pfnn_benchmark.hpp
//  pfnn_tvm
//
//  Created by haidonglan on 2021/7/21.
//

#ifndef pfnn_benchmark_hpp
#define pfnn_benchmark_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include "TVMInfer.hpp"
#include "EigenInfer.hpp"

using namespace Eigen;

class PFNN_Benchmark {
public:
    PFNN_Benchmark(int batch, std::string libPath, int inputVecLen, int hiddenVecLen, int outputVecLen);
    ~PFNN_Benchmark();
    
    void LoadSampleInputs(std::string dirpath);
    void LoadSampleOutputs(std::string dirpath);
    
    void BenchmarkInference(int repeats=100);
    void VerifyInference();
    void LoadModels(std::string dirpath);
private:
    TVMInfer *tvm_infer_engine;
    EigenInfer *eigen_infer_engine;
    TVMInfer *tvm_infer_engine_norm;
    EigenInfer *eigen_infer_engine_norm;
    int batch;
    int K1;
    int K2;
    int K3;
    int N3;
    float *Xmean;
    float *Xstd;
    float *Ymean;
    float *Ystd;
    float *sample_input;
    float *sample_output;
    int load_weights(TVM_DAT* buffer, int rows, int cols, const char* filename);
};


#endif /* pfnn_benchmark_hpp */
