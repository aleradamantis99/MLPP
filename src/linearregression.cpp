#include <linearregression.hpp>
#include "mlcommons.hpp"

/*********
* PUBLIC *
*********/
namespace ML
{
LinearRegression::LinearRegression(float learning_rate, size_t max_iter):
    learning_rate_(learning_rate), max_iter_(max_iter)
{}
LinearRegression::LinearRegression(float learning_rate): 
    learning_rate_(learning_rate) {}
LinearRegression::LinearRegression(size_t max_iter): 
    max_iter_(max_iter) {}

LinearRegression& LinearRegression::fit(const Array2D<float>& X, const std::vector<float>& y)
{
    auto [gd_w, gd_b] = gradient_descent(X, y, learning_rate_, max_iter_, linear_cost_gradient);
    w = std::move(gd_w);
    b = gd_b;
    n_features_ = X[0].size();
    return *this;
}

std::vector<float> LinearRegression::predict(const Array2D<float>& X)
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
float LinearRegression::predict(const std::vector<float>& x)
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
}// namespace ML