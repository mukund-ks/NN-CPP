#include "activation_fn.h"

#include <Eigen/Core>

RowMatrix SigmoidActivation::activate(const RowMatrix& input) const {
    return 1.0 / (1.0 + exp(-input.array()));
}

RowMatrix SigmoidActivation::gradient(const RowMatrix& input) const {
    RowMatrix val = activate(input);
    return val.array() * (1.0 - val.array());
}

RowMatrix ReLUActivation::activate(const RowMatrix& input) const {
    return input.array().max(0.0);
}

RowMatrix ReLUActivation::gradient(const RowMatrix& input) const {
    RowMatrix gradients(input.size());
    for (Eigen::Index i = 0; i < input.size(); i++) {
        gradients(i) = (input(i) > 0) ? 1 : 0;
    }
    return gradients;
}

RowMatrix LinearActivation::activate(const RowMatrix& input) const {
    return input;
}

RowMatrix LinearActivation::gradient(const RowMatrix& input) const {
    return RowMatrix::Ones(input.size());
}