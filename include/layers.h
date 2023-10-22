#pragma once
#include <Eigen/Dense>

#include "activation_fn.h"

typedef Eigen::MatrixXf Matrix;

class Layer {
   protected:
    ActivationFunction *activationFn;
    Matrix weights;
    Matrix biases;

   public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix &input) const = 0;
    virtual Matrix backward(const Matrix &input) const = 0;
};

class InputLayer : public Layer {
   public:
    InputLayer(int inputSize, int outputSize, ActivationFunction *activationFn);
    Matrix forward(const Matrix &input) const override;
    Matrix backward(const Matrix &input) const override;
};

class DenseLayer : public Layer {
   public:
    DenseLayer(int inputSize, int outputSize, ActivationFunction *activationFn);
    Matrix forward(const Matrix &input) const override;
    Matrix backward(const Matrix &input) const override;
};

class OutputLayer : public Layer {
   public:
    OutputLayer(int inputSize, int outputSize, ActivationFunction *activationFn);
    Matrix forward(const Matrix &input) const override;
    Matrix backward(const Matrix &input) const override;
};