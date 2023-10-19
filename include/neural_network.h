#pragma once
#include <Eigen/Dense>
#include <vector>

#include "layers.h"
#include "loss_fn.h"
#include "optimizers.h"

typedef Eigen::MatrixXf Matrix;

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