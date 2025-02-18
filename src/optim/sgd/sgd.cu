#include "cuda_fp16.h"

enum MomentumType {
    None,
    Classic,
    Nesterov,
};

enum WeightDecayType {
    WdNone,
    L2,
    Decoupled
};

struct SgdConfig {
    double lr;
    MomentumType momentum_type;
    double momentum;
    WeightDecayType weight_decay_type;
    double weight_decay;
};

template<typename T>
__device__ void sgd_update(
    const SgdConfig cfg,
    const size_t numel,
    T* param,
    T* velocity,
    const T* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    T weight_decay = cfg.weight_decay;
    T lr = cfg.lr;
    T momentum = cfg.momentum;

    T p = param[i];
    T g = grad[i];
    T v = velocity[i];

    if (cfg.weight_decay_type == L2) {
        g += weight_decay * p;
    }

    if (cfg.momentum_type == Classic) {
        v = g + momentum * v;
        g = v * lr;
    } else if (cfg.momentum_type == Nesterov) {
        v = g + momentum * v;
        g = (g + momentum * v) * lr;
    } else {
        g *= lr;
    }

    if (cfg.weight_decay_type == Decoupled) {
        g += weight_decay * lr * p;
    }

    velocity[i] = v;
    param[i] -= g;
}

#define SGD(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const SgdConfig cfg, \
    const size_t numel, \
    TYPENAME* param, \
    TYPENAME* velocity, \
    const TYPENAME* grad \
) { \
    sgd_update(cfg, numel, param, velocity, grad); \
}

SGD(__half, sgd_update_f16);
SGD(float, sgd_update_f32);
SGD(double, sgd_update_f64);
