#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

#include <array2D.hpp>
#include <utils.hpp>
#include <transformermixin.hpp>

namespace ML
{
class ZScoreNormalizer: public TransformerMixin<ZScoreNormalizer>
{
private:
    struct Statistics
    {
        float mean=0, stddev=0;
    };
    std::vector<Statistics> stats_;

    constexpr float norm_sample(float value, size_t feature) const;
    constexpr float inv_norm_sample(float norm_value, size_t feature) const;

    constexpr void update_feature_stats(const Array2D<float>& X, size_t feature);
public:
    ZScoreNormalizer() = default;

    ZScoreNormalizer& fit(const Array2D<float>& X);
    void transform(Array2D<float>& X) const;

    void inverse_transform(Array2D<float>& X) const;
};
}