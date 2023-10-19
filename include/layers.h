#pragma once
#include <Eigen/Dense>

#include "activation_fn.h"

typedef Eigen::MatrixXf Matrix;

class Layer {
   private:
    ActivationFunction *activationFunction;
    Matrix weights;
    Matrix biases;

   public:
    Layer(int inputSize, int outputSize);
    virtual Matrix forward(const Matrix &input) const;
    virtual Matrix backward(const Matrix &input) const;
};

class InputLayer : public Layer {
   public:
    InputLayer(int inputSize, int outputSize);
    Matrix forward(const Matrix &input) const override;
    Matrix backward(const Matrix &input) const override;
};

class DenseLayer : public Layer {
   public:
    DenseLayer(int inputSize, int outputSize);
    Matrix forward(const Matrix &input) const override;
    Matrix backward(const Matrix &input) const override;
};

class OutputLayer : public Layer {
   public:
    OutputLayer(int inputSize, int outputSize);
    Matrix forward(const Matrix &input) const override;
    Matrix backward(const Matrix &input) const override;
};