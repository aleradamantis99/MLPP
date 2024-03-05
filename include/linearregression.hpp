#pragma once
#include <cstddef>
#include <vector>
#include "array2D.hpp"
#include "regressormixin.hpp"
namespace ML
{
class LinearRegression: public RegressorMixin<LinearRegression>
{
public:
    LinearRegression() = default;
    LinearRegression(float learning_rate, size_t max_iter);
    LinearRegression(float learning_rate);
    LinearRegression(size_t max_iter);

    LinearRegression& fit(const Array2D<float>& X, const std::vector<float>& y);

    std::vector<float> predict(const Array2D<float>& X);
    float predict(const std::vector<float>& x);
private:
    float learning_rate_ = 0.001;
    size_t max_iter_ = 10000;

    size_t n_features_;

    std::vector<float> w{};
    float b;
};
}