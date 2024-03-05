#include <mlcommons.hpp>

namespace ML
{
namespace ranges = std::ranges;

float accuracy_score(const std::vector<int>& y, const std::vector<int>& y_pred)
{
    size_t correct_preds = 0; 
    for (auto [y_true_i, y_pred_i]: std::views::zip(y, y_pred))
    {
        if (y_true_i == y_pred_i)
        {
            correct_preds++;
        }
    }

    return correct_preds/static_cast<float>(y_pred.size());
}

float r2_score(const std::vector<float>& y, const std::vector<float>& y_pred)
{
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

float sigmoid(float z)
{
    z = std::clamp(z, -500.f, 500.f);
    return 1.f / (1.f+std::exp(-z));
}

std::pair<std::vector<float>, float> log_cost_gradient(const Array2D<float>& X, const std::vector<float>& y, const std::vector<float>& w, float b)
{
    size_t n = X.size();    
    std::vector<float> dj_dw(X[0].size(), 0);
    float dj_db = 0;

    for (size_t i=0; i<n; i++)
    {
        float f_wb = sigmoid(dot_product(w, X[i]) + b);
        float err = f_wb - y[i];
        ranges::transform(X[i], dj_dw, std::begin(dj_dw), [err, n](float x, float dw) { return err*x + dw; });
        dj_db += err;
    }
    ranges::transform(dj_dw, std::begin(dj_dw), [&n](float dw) { return dw/n; });
    return {std::move(dj_dw), dj_db/n};
}


}