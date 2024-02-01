#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

class ZScoreNormalizer
{
private:
    struct Statistics
    {
        float mean=0, stddev=0;
    };
    std::vector<Statistics> stats_;

    constexpr float norm_sample(float value, size_t feature) const;
    constexpr float inv_norm_sample(float norm_value, size_t feature) const;

    constexpr void update_feature_stats(const std::vector<std::vector<float>>& X, size_t feature);
public:
    ZScoreNormalizer() = default;
    constexpr ZScoreNormalizer& fit(const std::vector<std::vector<float>>& X);
    constexpr void transform(std::vector<std::vector<float>>& X) const;
    void fit_transform(std::vector<std::vector<float>>& X);

    void inverse_transform(std::vector<std::vector<float>>& X) const;
};