#pragma once

#include "array2D.hpp"
#include "mlcommons.hpp"
#include "crtp.hpp"
namespace ML
{
template <typename D>
class RegressorMixin: CRTP<D, RegressorMixin>
{
public:
    constexpr static EstimatorType estimator_type = EstimatorType::regressor;
    constexpr static bool requires_y = true;
    
    float score(const Array2D<float>& X, const std::vector<float>& y)
    {
        std::vector<float> y_pred = this->underlying().predict(X);
        return r2_score(y, y_pred);
    }
};

} // namespace ML