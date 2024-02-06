#include <zscorenormalizer.hpp>

/**********
* PRIVATE *
**********/
constexpr float ZScoreNormalizer::norm_sample(float value, size_t feature) const
{
    return (value-stats_[feature].mean)/stats_[feature].stddev;
}
constexpr float ZScoreNormalizer::inv_norm_sample(float norm_value, size_t feature) const
{
    return norm_value*stats_[feature].stddev + stats_[feature].mean;
}


constexpr void ZScoreNormalizer::update_feature_stats(const std::vector<std::vector<float>>& X, size_t feature)
{
    namespace ranges = std::ranges;
    float average = ranges::fold_left(X, 0.f, [feature](float prev, const std::vector<float>& x){ return prev+x[feature]; });
    float std_dev = std::sqrt(ranges::fold_left(X, 0.f, [feature, average](float prev, const std::vector<float>& x) { return prev + (x[feature] - average)*(x[feature] - average); }));
    stats_[feature] = Statistics{average, std_dev};
}

/*********
* PUBLIC *
*********/
constexpr ZScoreNormalizer& ZScoreNormalizer::fit(const std::vector<std::vector<float>>& X)
{   
    auto n_features = X[0].size();
    stats_.resize(n_features);

    for (size_t i=0; i<n_features; i++)
    {
        update_feature_stats(X, i);
    }
    return *this;
}
constexpr void ZScoreNormalizer::transform(std::vector<std::vector<float>>& X) const
{
    for (size_t i=0; i<stats_.size(); i++)
    {
        for (std::vector<float>& v: X)
        {
            v[i] = norm_sample(v[i], i);
        }
    }
}
void ZScoreNormalizer::fit_transform(std::vector<std::vector<float>>& X)
{
    auto n_features = X[0].size();
    stats_.resize(n_features);

    for (size_t i=0; i<n_features; i++)
    {
        update_feature_stats(X, i);
        for (std::vector<float>& v: X)
        {
            v[i] = norm_sample(v[i], i);
        }
    }
}

void ZScoreNormalizer::inverse_transform(std::vector<std::vector<float>>& X) const
{
    for (size_t i=0; i<stats_.size(); i++)
    {
        for (std::vector<float>& v: X)
        {
            v[i] = inv_norm_sample(v[i], i);
        }
    }
}