#pragma once
#include <vector>
#include <ranges>
#include <algorithm>
#include "array2D.hpp"
#include "utils.hpp"

namespace ML
{
namespace ranges = std::ranges;
std::pair<std::vector<float>, float> gradient_descent(const TwoDimensionalAccesible auto& X, const OneDimensionalAccesible auto& y, float alpha, size_t num_iters, auto gradient_function)
{
    

    float b = 0;
    std::vector<float> w(X[0].size(), 0);
    
    for (size_t i=0; i<num_iters; ++i)
    {
        auto [dj_dw, dj_db] = gradient_function(X, y, w, b);
        b = b - alpha*dj_db;

        ranges::transform(w, dj_dw, std::begin(w), [alpha](float w,  float dw) { return w-alpha*dw; });
    }

    return {std::move(w), b};
}

float linear_cost_function(const std::ranges::range auto& X, const ranges::range auto& y, const ranges::range auto& w, float b)
{
    auto n = X.size();
    float total_cost = 0.f;
    for (auto [X_i, y_i]: std::views::zip(X, y))
    {
        float y_pred = dot_product(w, X_i) + b;
        float diff = (y_i - y_pred)*(y_i - y_pred);
        total_cost += diff;
    }
    return total_cost/(2*n);
}

std::pair<std::vector<float>, float> linear_cost_gradient(const Array2D<float>& X, const std::vector<float>& y, const std::vector<float>& w, float b)
{
    size_t n = X.size();    
    std::vector<float> dj_dw(X[0].size(), 0);
    float dj_db = 0;

    for (size_t i=0; i<n; i++)
    {
        float f_wb = dot_product(w, X[i]) + b;
        float err = f_wb - y[i];
        ranges::transform(X[i], dj_dw, std::begin(dj_dw), [err, n](float x, float dw) { return err*x + dw; });
        dj_db += err;
    }
    ranges::transform(dj_dw, std::begin(dj_dw), [&n](float dw) { return dw/n; });
    return {std::move(dj_dw), dj_db/n};
}


}