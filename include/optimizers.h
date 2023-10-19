#pragma once
#include <Eigen/Dense>

typedef Eigen::MatrixXf Matrix;

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