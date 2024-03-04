#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>
#include "mlcommons.hpp"
#include "array2D.hpp"



namespace ML
{
class LogisticRegression
{
public:
    static constexpr size_t DEFAULT_MAX_ITER = 10000;
    static constexpr float DEFAULT_LEARNING_RATE = 0.001;
    static constexpr bool DEFAULT_MULTICLASS = false;

    #ifdef __cpp_designated_initializers
    struct ConstructorParams
    {
        float learning_rate = DEFAULT_LEARNING_RATE;
        size_t max_iter = DEFAULT_MAX_ITER;
        bool multiclass = DEFAULT_MULTICLASS;
    };
        //Used as LogisticRegression lr({.max_iter=1000, .multiclass=true});
    LogisticRegression(ConstructorParams p):
        learning_rate_(p.learning_rate),
        max_iter_(p.max_iter)
        //multiclass_(p.multiclass)
    {}
    #endif

    LogisticRegression() = default;


    LogisticRegression(float learning_rate, size_t max_iter):
        learning_rate_(learning_rate), max_iter_(max_iter)
    {}
    LogisticRegression(float learning_rate): 
        learning_rate_(learning_rate) {}
    LogisticRegression(size_t max_iter): 
        max_iter_(max_iter) {}

    void set_classes(const std::vector<int>& y)
    {
        labels_.resize(2);
        labels_[0] = y[0];
        for (int i: y)
        {
            if (i != labels_[0])
            {
                labels_[1] = i;
                break;
            }
        }
    }
    LogisticRegression& fit(const Array2D<float>& X, const std::vector<int>& y)
    {
        set_classes(y);
        std::vector<float> y_bin(y.size());
        namespace ranges = std::ranges;
        ranges::transform(y, std::begin(y_bin), [this](int i) { return i == this->labels_[0]? 0.f:1.f; });
        auto [gd_w, gd_b] = gradient_descent(X, y_bin, learning_rate_, max_iter_, log_cost_gradient);
        w = std::move(gd_w);
        b = gd_b;
        n_features_ = X[0].size();
        return *this;
    }

    std::vector<int> predict(const Array2D<float>& X)
    {
        assert(X[0].size()==n_features_);
        std::vector<int> y_pred;
        y_pred.reserve(X.size());
        for (auto sample: X)
        {
            float prob = std::round(find_prob(sample));

            y_pred.push_back(labels_[static_cast<size_t>(prob)]);
        }

        return y_pred;
    }
    std::vector<std::pair<float, float>> predict_proba(const Array2D<float>& X)
    {
        assert(X[0].size()==n_features_);
        std::vector<std::pair<float, float>> y_pred;
        y_pred.reserve(X.size());
        for (auto sample: X)
        {
            float prob = find_prob(sample);

            y_pred.emplace_back(1-prob, prob);
        }

        return y_pred;
    }
    
    float score(const Array2D<float>& X, const std::vector<int>& y)
    {
        std::vector<int> y_pred = predict(X);
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
private:

    float find_prob(std::span<const float> sample)
    {
        float pred = 0;
        for (size_t i=0; i<n_features_; i++)
        {
            pred += sample[i]*w[i];
        }
        pred += b;
        float prob = sigmoid(pred);
        return prob;
    }

    float learning_rate_ = DEFAULT_LEARNING_RATE;
    size_t max_iter_ = DEFAULT_MAX_ITER;

    size_t n_features_;
    std::vector<int> labels_;

    std::vector<float> w{};
    float b;
};
}