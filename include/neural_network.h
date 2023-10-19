#pragma once
#include <Eigen/Dense>
#include <vector>

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

class AdamOptimizer {
   private:
    float LR, beta1, beta2, epsilon;
    Matrix m, v;
    int t;
    void initializeMomentEstimates(const Matrix &weights);

   public:
    AdamOptimizer(float LR, float beta1, float beta2, float epsilon);
    void update(Matrix &weights, const Matrix &gradients, int iteration);
    void reset();
};

class NeuralNetwork {
   private:
    std::vector<Layer *> layers;
    Loss *lossFunction;
    AdamOptimizer optimizer;

   public:
    NeuralNetwork(Loss *lossFunction, AdamOptimizer optimizer);
    void addLayer(Layer *layer);
    void train(const Matrix &input, const Matrix &target, int epochs);
    Matrix validate(const Matrix &input, const Matrix &target);
    Matrix test(const Matrix &input);
    void save(const std::string &filename);
    void load(const std::string &filename);
};