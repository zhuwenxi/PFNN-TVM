//
//  BaseInfer.cpp
//  pfnn_tvm
//
//  Created by haidonglan on 2021/7/27.
//

#include "BaseInfer.hpp"
#include <string>
#include <vector>

BaseInfer::BaseInfer(const int batch,
        const int input_vector_len,
        const int layer1_channel,
        const int layer2_channel,
        const int output_vector_len) : batch(batch),
K1(input_vector_len),
N1(layer1_channel),
K2(layer1_channel),
N2(layer2_channel),
K3(layer2_channel),
N3(output_vector_len),
W(4),
Xmean(NULL),
Xstd(NULL),
Ymean(NULL),
Ystd(NULL)
{
    
}

BaseInfer::~BaseInfer() {
    if(Xmean) {
        free(Xmean);
    }
    if(Xstd) {
        free(Xstd);
    }
    if(Ymean) {
        free(Ymean);
    }
    if(Ystd) {
        free(Ystd);
    }
}

int BaseInfer::load_weights(TVM_DAT* buffer, int rows, int cols, FILE* f) {
    if (f == NULL) { fprintf(stderr, "Couldn't load file\n"); return -1; }
    // Weights are stored in transposed manner
    for (int y = 0; y < cols; y++) {
        for (int x = 0; x < rows; x++) {
            float item = 0.0;
            fread(&item, sizeof(float), 1, f);
            buffer[x * cols + y] = item;
        }
    }

    return 0;
}

int BaseInfer::load_weights(TVM_DAT* buffer, int rows, int cols, const char* filename) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) { fprintf(stderr, "Couldn't load file %s\n", filename); return -1; }
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

int BaseInfer::load_weights(TVM_DAT* buffer, int weights_num, int rows, int cols, const char* filename) {
    FILE *f = fopen(filename, "rb");
    fseek(f, 0, SEEK_SET);
    for (int i = 0; i < weights_num; ++i) {
        load_weights(buffer + i * rows * cols, rows, cols, f);
    }
    fclose(f);
    return 0;
}

void BaseInfer::LoadWeightsFromFiles(const char *dirpath) {
    char filename[256];
    // Load X / Y mean and std values
    Xmean = (float*)malloc(sizeof(float) * K1);
    Xstd = (float*)malloc(sizeof(float) * K1);
    
    Ymean = (float*)malloc(sizeof(float) * N3);
    Ystd = (float*)malloc(sizeof(float) * N3);
    
    sprintf(filename, "%s/Xmean.bin", dirpath);
    load_weights(Xmean, 1, K1, filename);
    sprintf(filename, "%s/Xstd.bin", dirpath);
    load_weights(Xstd, 1, K1, filename);
    sprintf(filename, "%s/Ymean.bin", dirpath);
    load_weights(Ymean, 1, N3, filename);
    sprintf(filename, "%s/Ystd.bin", dirpath);
    load_weights(Ystd, 1, N3, filename);
    
#if 1
    std::vector<TVM_DAT*> buffers;
    buffers.push_back((float*)malloc(sizeof(float) * W * K1 * K2));
    buffers.push_back((float*)malloc(sizeof(float) * W * K2 * K3));
    buffers.push_back((float*)malloc(sizeof(float) * W * K3 * N3));
    buffers.push_back((float*)malloc(sizeof(float) * W * K2));
    buffers.push_back((float*)malloc(sizeof(float) * W * K3));
    buffers.push_back((float*)malloc(sizeof(float) * W * N3));

    sprintf(filename, "%s/Nor_W0.bin", dirpath);
    load_weights(buffers[0], W, K1, K2, filename);
    sprintf(filename, "%s/Nor_W1.bin", dirpath);
    load_weights(buffers[1], W, K2, K3, filename);
    sprintf(filename, "%s/Nor_W2.bin", dirpath);
    load_weights(buffers[2], W, K3, N3, filename);
    sprintf(filename, "%s/Nor_b0.bin", dirpath);
    load_weights(buffers[3], W, 1, K2, filename);
    sprintf(filename, "%s/Nor_b1.bin", dirpath);
    load_weights(buffers[4], W, 1, K3, filename);
    sprintf(filename, "%s/Nor_b2.bin", dirpath);
    load_weights(buffers[5], W, 1, N3, filename);
    int ret = this->LoadModel(buffers[0], buffers[3], buffers[1], buffers[4], buffers[2], buffers[5]);
    if (ret != 0 ) {
        fprintf(stderr, "Load model @ %s failed, return code %d\n", dirpath, ret);
    }
    for(int i = 0; i < buffers.size(); ++i) {
        free(buffers[i]);
    } 
#else
    std::vector<std::string> model_wildcards;
    std::vector<TVM_DAT*> buffers;
    std::vector<std::pair<int, int>> shapes;
    if (use_normalized_models) {
        model_wildcards.push_back(std::string("%s/Nor_W0_%03i.bin"));
        model_wildcards.push_back(std::string("%s/Nor_b0_%03i.bin"));
        model_wildcards.push_back(std::string("%s/Nor_W1_%03i.bin"));
        model_wildcards.push_back(std::string("%s/Nor_b1_%03i.bin"));
        model_wildcards.push_back(std::string("%s/Nor_W2_%03i.bin"));
        model_wildcards.push_back(std::string("%s/Nor_b2_%03i.bin"));
    } else {
        model_wildcards.push_back(std::string("%s/W0_%03i.bin"));
        model_wildcards.push_back(std::string("%s/b0_%03i.bin"));
        model_wildcards.push_back(std::string("%s/W1_%03i.bin"));
        model_wildcards.push_back(std::string("%s/b1_%03i.bin"));
        model_wildcards.push_back(std::string("%s/W2_%03i.bin"));
        model_wildcards.push_back(std::string("%s/b2_%03i.bin"));
    }
    
    buffers.push_back((float*)malloc(sizeof(float) * K1 * K2));
    buffers.push_back((float*)malloc(sizeof(float) * K2));
    buffers.push_back((float*)malloc(sizeof(float) * K2 * K3));
    buffers.push_back((float*)malloc(sizeof(float) * K3));
    buffers.push_back((float*)malloc(sizeof(float) * K3 * N3));
    buffers.push_back((float*)malloc(sizeof(float) * N3));
    shapes.push_back(std::make_pair(K1, K2));
    shapes.push_back(std::make_pair(1, K2));
    shapes.push_back(std::make_pair(K2, K3));
    shapes.push_back(std::make_pair(1, K3));
    shapes.push_back(std::make_pair(K3, weights_layer3_stride));
    shapes.push_back(std::make_pair(1, weights_layer3_stride));
    for (int m = 0; m < model_count; ++m) {
        for(int i = 0; i < model_wildcards.size(); ++i) {
            sprintf(filename, model_wildcards[i].c_str(), dirpath, m);
            load_weights(buffers[i], shapes[i].first, shapes[i].second, filename);
        }
        int ret = this->LoadModel(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],    buffers[5]);
        if (ret != 0 ) {
            fprintf(stderr, "Add model %d @ %s failed, return code %d\n", m, dirpath, ret);
        }
    }
    for(int i = 0; i < buffers.size(); ++i) {
        free(buffers[i]);
    }
#endif
}

TVM_DAT BaseInfer::GetYmean(int vector_index) {
    if (vector_index >= 0 && vector_index <= K1 - 1) {
        return Ymean[vector_index];
    } else {
        fprintf(stderr, "Get Ymean vector index %d out of range %d--%d\n", vector_index, 0, K1 - 1);
        return -914.914914f;
    }
}
