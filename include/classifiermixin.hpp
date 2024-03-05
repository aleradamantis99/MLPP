#pragma once
#include "array2D.hpp"
#include "mlcommons.hpp"
#include "crtp.hpp"
namespace ML
{
template <typename D>
class ClassifierMixin: CRTP<D, ClassifierMixin>
{
public:
    constexpr static EstimatorType estimator_type = EstimatorType::classifier;
    constexpr static bool requires_y = true;
    
    float score(const Array2D<float>& X, const std::vector<int>& y)
    {
        std::vector<int> y_pred = this->underlying().predict(X);
        return accuracy_score(y, y_pred);
    }
};

} // namespace ML