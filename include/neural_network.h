#pragma once
#include <Eigen/Dense>
#include <vector>

#include "layers.h"
#include "loss_fn.h"
#include "optimizers.h"

typedef Eigen::MatrixXf Matrix;
typedef std::vector<std::pair<Eigen::MatrixXf, std::vector<int>>> Set;

class NeuralNetwork {
   private:
    std::vector<Layer *> layers;
    Loss *lossFunction;
    AdamOptimizer optimizer;

   public:
    NeuralNetwork(Loss *lossFunction, AdamOptimizer optimizer);
    void addLayer(Layer *layer);
    void train(const Set &dataset, int epochs);
    Matrix validate(const Set &dataset);
    Matrix test(const Set &dataset);
    void save(const std::string &filename);
    void load(const std::string &filename);
};