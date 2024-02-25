#pragma once
#include <cstddef>
#include <vector>
#include "mlcommons.hpp"
#include "array2D.hpp"

namespace ML
{
class LinearRegression
{
public:
    LinearRegression() = default;
    LinearRegression(float learning_rate, size_t max_iter):
        learning_rate_(learning_rate), max_iter_(max_iter)
    {}
    LinearRegression(float learning_rate): 
        learning_rate_(learning_rate) {}
    LinearRegression(size_t max_iter): 
        max_iter_(max_iter) {}

    LinearRegression& fit(const Array2D<float>& X, const std::vector<float>& y)
    {
        auto [gd_w, gd_b] = gradient_descent(X, y, learning_rate_, max_iter_, linear_cost_gradient);
        w = std::move(gd_w);
        b = gd_b;
        n_features_ = X[0].size();
        return *this;
    }

    std::vector<float> predict(const Array2D<float>& X)
    {
        assert(X[0].size()==n_features_);
        std::vector<float> y_pred;
        y_pred.reserve(X.size());
        for (auto sample: X)
        {
            float pred = 0;
            for (size_t i=0; i<n_features_; i++)
            {
                pred += sample[i]*w[i];
            }
            pred += b;
            y_pred.push_back(pred);
        }

        return y_pred;
    }
    float predict(const std::vector<float>& x)
    {
        assert(x.size()==n_features_);
        float pred = 0;
        for (size_t i=0; i<n_features_; i++)
        {
            pred += x[i]*w[i];
        }
        pred += b;
        
        return pred;
    }
    
    float score(const Array2D<float>& X, const std::vector<float>& y)
    {
        std::vector<float> y_pred = predict(X);
        //u=((y_true - y_pred)** 2).sum()
        float u=0;
        for (auto [y_true_i, y_pred_i]: std::views::zip(y, y_pred))
        {
            u+=(y_true_i-y_pred_i)*(y_true_i-y_pred_i);
        }
        //v=((y_true - y_true.mean()) ** 2).sum()
        float v = 0;
        int mean = std::ranges::fold_left(y.begin(), y.end(), 0, std::plus<float>()) / y.size(); 
        
        for (float y_true_i: y)
        {
            v += (y_true_i-mean)*(y_true_i-mean);
        }
        float R2 = 1-(u/v);
        return R2;
    }
private:
    float learning_rate_ = 0.001;
    size_t max_iter_ = 10000;

    size_t n_features_;

    std::vector<float> w{};
    float b;
};
}