//
//  eigen_infer.cpp
//  pfnn_tvm
//
//  Created by haidonglan on 2021/7/21.
//

#include "EigenInfer.hpp"
#include <iostream>

EigenInfer::EigenInfer(const int batch,
                   const int input_vector_len,
                   const int layer1_channel,
                   const int layer2_channel,
                   const int output_vector_len) : BaseInfer(batch, input_vector_len, layer1_channel, layer2_channel, output_vector_len), w1(), w2(), w3(),b1(),b2(),b3()
{
    this->inputTensor = Eigen::ArrayXXf(batch, K1);
}


int EigenInfer::LoadModel(TVM_DAT *weights_layer1, TVM_DAT *bias_layer1,
                         TVM_DAT *weights_layer2, TVM_DAT *bias_layer2,
                         TVM_DAT *weights_layer3, TVM_DAT *bias_layer3) {
    for (int i = 0; i < W; ++i) {
        // printf("Eigen loading model %d\n", i);
        Eigen::ArrayXXf w1t = Eigen::Map<Eigen::ArrayXXf>(weights_layer1 + i * N1 * K1, N1, K1);
        Eigen::ArrayXXf w2t = Eigen::Map<Eigen::ArrayXXf>(weights_layer2 + i * N2 * K2, N2, K2);
        Eigen::ArrayXXf w3t = Eigen::Map<Eigen::ArrayXXf>(weights_layer3 + i * N3 * K3, N3, K3);
        Eigen::ArrayXf b1line = Eigen::Map<Eigen::ArrayXf>(bias_layer1 + i * N1, N1);
        Eigen::ArrayXf b2line = Eigen::Map<Eigen::ArrayXf>(bias_layer2 + i * N2, N2);
        Eigen::ArrayXf b3line = Eigen::Map<Eigen::ArrayXf>(bias_layer3 + i * N3, N3);
    
        this->w1.push_back(w1t);
        this->w2.push_back(w2t);
        this->w3.push_back(w3t);
        this->b1.push_back(b1line);
        this->b2.push_back(b2line);
        this->b3.push_back(b3line);
    }

#ifdef EIGEN_INFER_BATCH_MODE
    Eigen::ArrayXXf w1_c(N1, K1 * W);
    w1_c << w1[0], w1[1], w1[2], w1[3];
    this->w1_combo = w1_c.transpose();
    Eigen::ArrayXXf w2_c(N2, K2 * W);
    w2_c << w2[0], w2[1], w2[2], w2[3];
    this->w2_combo = w2_c.transpose();
    Eigen::ArrayXXf w3_c(N3, K3 * W);
    w3_c << w3[0], w3[1], w3[2], w3[3];
    this->w3_combo = w3_c.transpose();
    Eigen::ArrayXf b1_c(N1 * W);
    b1_c << b1[0], b1[1], b1[2], b1[3];
    this->b1_combo = b1_c;
    Eigen::ArrayXf b2_c(N2 * W);
    b2_c << b2[0], b2[1], b2[2], b2[3];
    this->b2_combo = b2_c;
    Eigen::ArrayXf b3_c(N3 * W);
    b3_c << b3[0], b3[1], b3[2], b3[3];
    this->b3_combo = b3_c;
#endif
    return 0;
}

static void ELU(Eigen::ArrayXf &x) { x = x.max(0) + x.min(0).exp() - 1; }
static void ELU(Eigen::ArrayXXf &x) { x = x.max(0) + x.min(0).exp() - 1; }

Eigen::ArrayXXf MakeCoeff(std::vector<float>& phases) {
    Eigen::ArrayXXf Coeff(phases.size(), 4);
    for (int i = 0; i < phases.size(); ++i) {
        int pindex[4];
        float P = phases[i];
        float mu = fmod((P / (2*M_PI)) * 4, 1.0);
        float mu_cubic = mu * mu * mu;
        float mu_square = mu * mu;
        pindex[1] = (int)((P / (2*M_PI)) * 4);
        pindex[0] = ((pindex[1]+3) % 4);
        pindex[2] = ((pindex[1]+1) % 4);
        pindex[3] = ((pindex[1]+2) % 4);
        int coeff_order[4];
        for (int j = 0; j < 4; ++j) {
            coeff_order[pindex[j]] = j;
        }
        float coeff_line[4];
        coeff_line[0] = -0.5 * mu_cubic + mu_square - 0.5 * mu;
        coeff_line[1] = 1.5 * mu_cubic - 2.5 * mu_square + 1;
        coeff_line[2] = -1.5 * mu_cubic + 2 * mu_square + 0.5 * mu;
        coeff_line[3] = 0.5 * mu_cubic - 0.5 * mu_square;
        for (int j = 0; j < 4; ++j) {
            Coeff(i, j) = coeff_line[coeff_order[j]];
        }
    }
    return Coeff;
}

static void cubic(Eigen::ArrayXf  &o, const Eigen::ArrayXf &y0, const Eigen::ArrayXf &y1, const Eigen::ArrayXf &y2, const Eigen::ArrayXf &y3, float mu) {
    o = (
      (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
      (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
      (-0.5*y0+0.5*y2)*mu + 
      (y1));
}

static void cubic(const Eigen::ArrayXXf & Coeff, Eigen::ArrayXXf &o, const Eigen::ArrayXXf &y0, const Eigen::ArrayXXf &y1, const Eigen::ArrayXXf &y2, const Eigen::ArrayXXf &y3, float mu) {
    o = Coeff(0, 0) * y0 + Coeff(0, 1) * y1 + Coeff(0, 2) * y2 + Coeff(0, 3) * y3;
}
static void cubic(Eigen::ArrayXXf &o, const Eigen::ArrayXXf &y0, const Eigen::ArrayXXf &y1, const Eigen::ArrayXXf &y2, const Eigen::ArrayXXf &y3, float mu) {
    o = (
      (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
      (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
      (-0.5*y0+0.5*y2)*mu + 
      (y1));
}

Eigen::ArrayXXf TensorCubicInterpolate(const Eigen::ArrayXXf& tensor, const Eigen::ArrayXXf& coeff) {
    Eigen::ArrayXXf t1(tensor.rows(), tensor.cols() * 4);
    for(int i = 0; i < tensor.rows(); ++i) {
        t1.block(i,                 0, 1, tensor.cols()) = coeff(i, 0) * tensor.row(i);
        t1.block(i,     tensor.cols(), 1, tensor.cols()) = coeff(i, 1) * tensor.row(i);
        t1.block(i, tensor.cols() * 2, 1, tensor.cols()) = coeff(i, 2) * tensor.row(i);
        t1.block(i, tensor.cols() * 3, 1, tensor.cols()) = coeff(i, 3) * tensor.row(i);
    }
    return t1;
}

Eigen::ArrayXXf TensorCubicInterpolateBias(const Eigen::ArrayXf& tensor, const Eigen::ArrayXXf& coeff) {
    size_t elem_len = tensor.rows() / 4;
    Eigen::ArrayXXf t1(coeff.rows(), elem_len);
    for(int i = 0; i < coeff.rows(); ++i) {
        Eigen::ArrayXXf ti(1, elem_len);
        ti = coeff(i, 0) * tensor.block(0, 0, elem_len, 1).transpose();
        ti = ti + coeff(i, 1) * tensor.block(elem_len, 0, elem_len, 1).transpose();
        ti = ti + coeff(i, 2) * tensor.block(elem_len * 2, 0, elem_len, 1).transpose();
        ti = ti + coeff(i, 3) * tensor.block(elem_len * 3, 0, elem_len, 1).transpose();
        t1.block(i, 0, 1, elem_len) = ti;
    }
    return t1;
}

#ifdef EIGEN_INFER_BATCH_MODE
int EigenInfer::Forward(std::vector<float> phases) {
    Eigen::ArrayXXf coeff = MakeCoeff(phases);
    Eigen::ArrayXXf b1 = TensorCubicInterpolateBias(this->b1_combo, coeff);
    Eigen::ArrayXXf b2 = TensorCubicInterpolateBias(this->b2_combo, coeff);
    Eigen::ArrayXXf b3 = TensorCubicInterpolateBias(this->b3_combo, coeff);

    Eigen::ArrayXXf t0 = TensorCubicInterpolate(this->inputTensor, coeff);
    Eigen::ArrayXXf H0 = (t0.matrix() * w1_combo.matrix()).array() + b1;
    ELU(H0);
    Eigen::ArrayXXf t1 = TensorCubicInterpolate(H0, coeff);
    Eigen::ArrayXXf H1 = (t1.matrix() * w2_combo.matrix()).array() + b2; 
    ELU(H1);
    Eigen::ArrayXXf t2 = TensorCubicInterpolate(H1, coeff);
    Eigen::ArrayXXf H2 = (t2.matrix() * w3_combo.matrix()).array() + b3; 

    outputTensor = H2;
    return 0;
}
#else
int EigenInfer::Forward(std::vector<float> phases) {
    Eigen::ArrayXXf W0p;
    Eigen::ArrayXXf W1p;
    Eigen::ArrayXXf W2p;
    Eigen::ArrayXf b0p;
    Eigen::ArrayXf b1p;
    Eigen::ArrayXf b2p;
    Eigen::ArrayXf H0;
    Eigen::ArrayXf H1;
    Eigen::ArrayXf H2;
    Eigen::ArrayXXf Y(batch, N3);
    
    for (int i = 0; i < phases.size(); ++i) {
        Eigen::ArrayXf Xp = this->inputTensor.row(i);
        float P = phases[i];
        float pamount = fmod((P / (2*M_PI)) * 4, 1.0);
        int pindex_1 = (int)((P / (2*M_PI)) * 4);
        int pindex_0 = ((pindex_1+3) % 4);
        int pindex_2 = ((pindex_1+1) % 4);
        int pindex_3 = ((pindex_1+2) % 4);
        
        //Eigen::ArrayXXf coeff = MakeCoeff(phases);
        cubic(W0p, w1[pindex_0], w1[pindex_1], w1[pindex_2], w1[pindex_3], pamount);
        cubic(W1p, w2[pindex_0], w2[pindex_1], w2[pindex_2], w2[pindex_3], pamount);
        cubic(W2p, w3[pindex_0], w3[pindex_1], w3[pindex_2], w3[pindex_3], pamount);
        cubic(b0p, b1[pindex_0], b1[pindex_1], b1[pindex_2], b1[pindex_3], pamount);
        cubic(b1p, b2[pindex_0], b2[pindex_1], b2[pindex_2], b2[pindex_3], pamount);
        cubic(b2p, b3[pindex_0], b3[pindex_1], b3[pindex_2], b3[pindex_3], pamount);
        H0 = (W0p.matrix() * Xp.matrix()).array() + b0p; 
        ELU(H0);
        H1 = (W1p.matrix() * H0.matrix()).array() + b1p; 
        ELU(H1);
        Y.row(i) = (W2p.matrix() * H1.matrix()).array() + b2p;
    }
    outputTensor = Y;
  return 0;
}
#endif

void EigenInfer::SetInput(int vector_index, TVM_DAT value) {
    int rows = vector_index / this->N1;
    int cols = vector_index % this->N1;
    inputTensor(rows, cols) = value;
}

void EigenInfer::SetInput(int index_in_batch, int vector_index, TVM_DAT value) {
    inputTensor(index_in_batch, vector_index) = value;
}


TVM_DAT EigenInfer::GetOutput(int index_in_batch, int vector_index) {
    return this->outputTensor(index_in_batch, vector_index);
}
