#pragma once
#include <Eigen/Dense>

typedef Eigen::MatrixXf Matrix;

class Loss {
   public:
    virtual Matrix compute(const Matrix &predicted, const Matrix &actual) const = 0;
    virtual Matrix gradient(const Matrix &predicted, const Matrix &actual) const = 0;
};

class MSE : public Loss {
   public:
    Matrix compute(const Matrix &predicted, const Matrix &actual) const override;
    Matrix gradient(const Matrix &predicted, const Matrix &actual) const override;
};

class SCCE : public Loss {
   public:
    Matrix compute(const Matrix &predicted, const Matrix &actual) const override;
    Matrix gradient(const Matrix &predicted, const Matrix &actual) const override;
};