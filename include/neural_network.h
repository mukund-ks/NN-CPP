#pragma once
#include <Eigen/Dense>
#include <vector>

class ActivationFunction {
   public:
    virtual Eigen::MatrixXf activate(const Eigen::MatrixXf &input) const = 0;
    virtual Eigen::MatrixXf gradient(const Eigen::MatrixXf &output) const = 0;
};

class SigmoidActivation : public ActivationFunction {
   public:
    Eigen::MatrixXf activate(const Eigen::MatrixXf &input) const override;
    Eigen::MatrixXf gradient(const Eigen::MatrixXf &output) const override;
};

class ReLUActivation : public ActivationFunction {
   public:
    Eigen::MatrixXf activate(const Eigen::MatrixXf &input) const override;
    Eigen::MatrixXf gradient(const Eigen::MatrixXf &output) const override;
};

class LinearActivation : public ActivationFunction {
   public:
    Eigen::MatrixXf activate(const Eigen::MatrixXf &input) const override;
    Eigen::MatrixXf gradient(const Eigen::MatrixXf &output) const override;
};

template <typename ActivationType>
class Layer {
   public:
    Layer(int inputSize, int outputSize);
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf &input) const;
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf &input) const;

   private:
    ActivationType *activationFunction;
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;
};

template <typename ActivationType>
class InputLayer : public Layer<ActivationType> {
   public:
    InputLayer(int inputSize, int outputSize);
    Eigen::MatrixXf forward(const Eigen::MatrixXf &input) const override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf &input) const override;
};

template <typename ActivationType>
class DenseLayer : public Layer<ActivationType> {
   public:
    DenseLayer(int inputSize, int outputSize);
    Eigen::MatrixXf forward(const Eigen::MatrixXf &input) const override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf &input) const override;
};

template <typename ActivationType>
class OutputLayer : public Layer<ActivationType> {
   public:
    OutputLayer(int inputSize, int outputSize);
    Eigen::MatrixXf forward(const Eigen::MatrixXf &input) const override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf &input) const override;
};