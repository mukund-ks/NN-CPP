#pragma once
#include <Eigen/Dense>

typedef Eigen::RowVectorXf RowMatrix;

class ActivationFunction {
   public:
    virtual RowMatrix activate(const RowMatrix &input) const = 0;
    virtual RowMatrix gradient(const RowMatrix &input) const = 0;
};

class SigmoidActivation : public ActivationFunction {
   public:
    RowMatrix activate(const RowMatrix &input) const override;
    RowMatrix gradient(const RowMatrix &input) const override;
};

class ReLUActivation : public ActivationFunction {
   public:
    RowMatrix activate(const RowMatrix &input) const override;
    RowMatrix gradient(const RowMatrix &input) const override;
};

class LinearActivation : public ActivationFunction {
   public:
    RowMatrix activate(const RowMatrix &input) const override;
    RowMatrix gradient(const RowMatrix &input) const override;
};