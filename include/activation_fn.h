#pragma once
#include <Eigen/Dense>

typedef Eigen::MatrixXf Matrix;

class ActivationFunction {
   public:
    virtual Matrix activate(const Matrix &input) const = 0;
    virtual Matrix gradient(const Matrix &output) const = 0;
};

class SigmoidActivation : public ActivationFunction {
   public:
    Matrix activate(const Matrix &input) const override;
    Matrix gradient(const Matrix &output) const override;
};

class ReLUActivation : public ActivationFunction {
   public:
    Matrix activate(const Matrix &input) const override;
    Matrix gradient(const Matrix &output) const override;
};

class LinearActivation : public ActivationFunction {
   public:
    Matrix activate(const Matrix &input) const override;
    Matrix gradient(const Matrix &output) const override;
};