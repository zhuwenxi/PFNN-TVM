//
//  pfnn_benchmark.cpp
//  pfnn_tvm
//
//  Created by haidonglan on 2021/7/21.
//

#include "pfnn_benchmark.hpp"
#include "timer.h"

#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstdlib>
#include <stdarg.h>
#include <time.h>
#include <stdio.h>
//#define DISABLE_EIGEN
PFNN_Benchmark::PFNN_Benchmark(int batch, std::string libPath, int inputVecLen, int hiddenVecLen, int outputVecLen) : tvm_infer_engine(NULL), batch(batch), K1(inputVecLen), K2(hiddenVecLen), K3(hiddenVecLen), N3(outputVecLen) {
#ifndef DISABLE_TVM
    tvm_infer_engine = new TVMInfer(libPath, batch, K1, K2, K3, N3);
#endif
#ifndef DISABLE_EIGEN
    eigen_infer_engine = new EigenInfer(batch, K1, K2, K3, N3);
#endif
}

PFNN_Benchmark::~PFNN_Benchmark(){
    if (tvm_infer_engine != NULL) {
        delete tvm_infer_engine;
    }
    if (eigen_infer_engine != NULL) {
        delete eigen_infer_engine;
    }
    if (sample_input != NULL) {
        free(sample_input);
    }
    if (sample_output != NULL) {
        free(sample_output);
    }
}

int PFNN_Benchmark::load_weights(TVM_DAT* buffer, int rows, int cols, const char* filename) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "Couldn't load file %s\n", filename);
        return -1;
    }
    // Weights are stored in transposed manner
    for (int y = 0; y < cols; y++) {
        for (int x = 0; x < rows; x++) {
            float item = 0.0;
            fread(&item, sizeof(float), 1, f);
            buffer[x * cols + y] = item;
        }
    }
    fclose(f);
    return 0;
}

void PFNN_Benchmark::LoadSampleInputs(const std::string dirpath) {
    char filename[256];
    
    sample_input =(float*)malloc(sizeof(float) * batch * K1);
    sprintf(filename, "%s/inputs.bin", dirpath.c_str());
    load_weights(sample_input, batch, K1, filename);
    
    for(int j = 0; j < batch; ++j){
        for(int i = 0; i < K1; ++i) {
#ifndef DISABLE_TVM
            tvm_infer_engine->SetInput(j, i, sample_input[j * K1 + i]);
#endif            
#ifndef DISABLE_EIGEN
            eigen_infer_engine->SetInput(j, i, sample_input[j * K1 + i]);
#endif
        }
    }
}

void PFNN_Benchmark::LoadSampleOutputs(const std::string dirpath) {
    char filename[256];
    sample_output =(float*)malloc(sizeof(float) * batch * N3);
    sprintf(filename, "%s/outputs.bin", dirpath.c_str());
    load_weights(sample_output, batch, N3, filename);
}

void PFNN_Benchmark::LoadModels(const std::string dirpath) {
    printf("Loading models...\n");
#ifndef DISABLE_EIGEN
    eigen_infer_engine->LoadWeightsFromFiles(dirpath.c_str());
#endif
#ifndef DISABLE_TVM
    tvm_infer_engine->LoadWeightsFromFiles(dirpath.c_str());
#endif
    printf("Done.\n");
}

void setRandPhases(std::vector<float>& phases) {
    srand(static_cast <unsigned> (time(0)));
    for (int i = 0; i < phases.size(); ++i) {
        phases[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2.0 * M_PI;
        // phases[i] = 1.5;
    }
}

//#define PFNN_LOG_EVERY_INFERENCE
void PFNN_Benchmark::BenchmarkInference(int repeats) {
    Timer computeTime;
    double cumtime = 0.f;
    std::vector<float> phases;
    phases.resize(batch);
    setRandPhases(phases);
#ifndef DISABLE_EIGEN
    eigen_infer_engine->Forward(phases);
    for(int i = 0; i < repeats; ++i) {
        setRandPhases(phases);
        computeTime.start();
        eigen_infer_engine->Forward(phases);
        computeTime.stop();
#ifdef PFNN_LOG_EVERY_INFERENCE
        printf("Eigen compute time %lfms\n", computeTime.elapsedMilliseconds());
#endif
        cumtime += computeTime.elapsedMilliseconds();
    }
#endif
    double cumtime_tvm = 0.f;
#ifndef DISABLE_TVM
    tvm_infer_engine->Forward(phases);
    for(int i = 0; i < repeats; ++i) {
        setRandPhases(phases);
        computeTime.start();
        tvm_infer_engine->Forward(phases);
        computeTime.stop();
#ifdef PFNN_LOG_EVERY_INFERENCE
        printf("TVM compute time %lfms\n", computeTime.elapsedMilliseconds());
#endif
        cumtime_tvm += computeTime.elapsedMilliseconds();
    }
#endif
#ifndef DISABLE_EIGEN
    printf("Eigen mean time %lf\n", cumtime / (double) repeats);
#endif
#ifndef DISABLE_TVM
    printf("TVM mean time %lf\n", cumtime_tvm / (double) repeats);
#endif
}

void PFNN_Benchmark::VerifyInference() {
    // Setup Timers
    Timer fullTime;
    Timer computeTime;
    
    // Run forward
    double cumtime = 0.f;
    std::vector<float> phases;
    phases.resize(batch);
    setRandPhases(phases);
    tvm_infer_engine->Forward(phases);
    eigen_infer_engine->Forward(phases);
    // Get output
    double cum_diff = 0.f;
    double cum_diff_tvm = 0.f;
    double max_diff = 0.f;
    double max_diff_tvm = 0.f;
    for (int i = 0; i < batch; ++i) {
        for(int j = 0; j < N3; ++j){
            float a = this->sample_output[N3 * i + j];
            float b = eigen_infer_engine->GetOutput(i, j);
            float c = tvm_infer_engine->GetOutput(i, j);
#ifdef COMPARE_SAMPLE_OUTPUTS
            if(fabs(a - b) > 1e-5 * fmax(fabs(a), fabs(b))){
                cum_diff += fabs(a - b);
                max_diff = fmax(max_diff, fabs(a-b));
                printf("Eigen diff (%d, %d) %f vs %f\n", i, j, a, b);
            }
#endif
            if(fabs(b - c) > 1e-3 * fmax(fabs(b), fabs(c))){
                cum_diff_tvm += fabs(b - c);
                max_diff_tvm = fmax(max_diff_tvm, fabs(b-c));
                printf("TVM diff (%d, %d) %f vs %f\n", i, j, b, c);
            }
        }
    }
    //printf("cumdiff %lf maxdiff %lf\n", cum_diff, max_diff);
    //printf("TVM cumdiff %lf maxdiff %lf\n", cum_diff_tvm, max_diff_tvm);
}
