#pragma once
#include <vector>
#include <variant>
#include <array2D.hpp>
#include <transformermixin.hpp>
#include <utils.hpp>
#include <algorithm>

namespace ML
{
class PolynomialFeatures : public TransformerMixin<PolynomialFeatures>
{
private:
    using degree_type = int;
    using degree_range_type = std::pair<degree_type, degree_type>;
    std::pair<int, int> degree_ = DEFAULT_DEGREE;
    bool interaction_only_ = false;
    bool include_bias_ = false;

    size_t n_features_ = 0;
    size_t n_features_out_;
    size_t n_out_full_;

    static constexpr size_t combinations_(size_t n_features, const degree_type& min_degree, const degree_type& max_degree, bool interaction_only, bool include_bias)
    {
        size_t combinations;
        if (interaction_only)
        {
            size_t start = std::max(1, min_degree);
            size_t end = std::min((size_t)max_degree, n_features);
            
            combinations = std::ranges::fold_left(std::views::iota(start, end+1), 0, [n_features](size_t prev, size_t next_value) { return prev + binomial_coefficient(n_features, next_value); });
        }
        else
        {
            combinations = binomial_coefficient(n_features+max_degree, (size_t)max_degree)-1;
            if (min_degree > 0)
            {
                auto d = min_degree - 1;
                combinations -= binomial_coefficient(n_features + d, d) - 1;
            }
        }

        if (include_bias)
        {
            combinations += 1;
        }

        return combinations;
    }


public:
    static constexpr std::pair<int, int> DEFAULT_DEGREE = {0, 2};
    static constexpr bool DEFAULT_INTERACTION_ONLY = false;
    static constexpr bool DEFAULT_INCLUDE_BIAS = true;



    PolynomialFeatures() = default;
    #ifdef __cpp_designated_initializers
    struct ConstructorParams
    {
        std::variant<int, std::pair<int, int>> degree = 2;
        /*
        int degree=-1;
        std::pair<int, int> degree_range = DEFAULT_DEGREE;
        //Then check, if degree -1 use degree_range, if not, use degree.
        */
        bool interaction_only = DEFAULT_INTERACTION_ONLY;
        bool include_bias = DEFAULT_INCLUDE_BIAS;
    };
    PolynomialFeatures(ConstructorParams p):
        interaction_only_(p.interaction_only),
        include_bias_(p.include_bias)
    {
        std::visit([this](auto&& d)
        {
            using T = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<T, int>)
            {
                degree_ = {0, d};
            }
            else
            {
                degree_ = d;
            }
        }, p.degree);
    }
    #endif
    explicit PolynomialFeatures(int degree, bool interaction_only=DEFAULT_INTERACTION_ONLY, bool include_bias=DEFAULT_INCLUDE_BIAS):
        degree_(degree, degree),
        interaction_only_(interaction_only),
        include_bias_(include_bias)
    {}
    explicit PolynomialFeatures(std::pair<int, int> degree, bool interaction_only=DEFAULT_INTERACTION_ONLY, bool include_bias=DEFAULT_INCLUDE_BIAS):
        degree_(degree),
        interaction_only_(interaction_only),
        include_bias_(include_bias)
    {}

    constexpr PolynomialFeatures& fit(const Array2D<float>& X)
    {
        n_features_ = X[0].size();
        /*n_features_out_ = binomial_coefficient(n_features_+degree_, degree_);
        std::ranges::next_permutation*/

        auto [min_degree, max_degree] = degree_;
        if (not (min_degree >=0 and min_degree <= max_degree))
        {
            throw std::invalid_argument(std::format("Invalid degree: degrees should be positive and min_degree ({}) <= max_degree ({})", min_degree, max_degree));
        }
        else if (max_degree == 0 and not include_bias_)
        {
            throw std::invalid_argument("Setting both min_degree and max_degree to zero and include_bias to False would result in an empty output array.");
        }

        n_features_out_ = combinations_(
            n_features_,
            min_degree,
            max_degree,
            interaction_only_,
            include_bias_
        );
        n_out_full_ = combinations_(
            n_features_,
            0,
            max_degree,
            interaction_only_,
            include_bias_
        );

        return *this;
    }
    [[nodiscard]] Array2D<float> transform(const Array2D<float>& X) const
    {
        namespace ranges = std::ranges;
        if (n_features_ == 0)
        {
            throw std::logic_error("Estimator is not fitted or fit data was empty");
        }

        auto [min_degree, max_degree] = degree_;

        size_t n_samples = X.size(), n_features = X[0].size();
        Array2D<float> XP(n_samples, n_out_full_);
        size_t current_col = 0;
        if (include_bias_)
        {
            ranges::fill(XP[][0], 1);
            current_col = 1;
        }

        if (max_degree == 0)
        {
            return XP;
        }

        for (auto&& [row, og_row]: std::views::zip(XP, X))
        {
            //ranges::copy(og_row, row | std::views::drop(current_col));
            std::copy(std::begin(og_row), std::end(og_row), std::begin(row)+current_col);
        }

        std::vector<size_t> index(n_features+1);
        ranges::iota(index, current_col);
        current_col += n_features;
        std::vector<size_t> new_index;
        new_index.reserve(n_features+1);
        for (int i=2; i<=max_degree; i++)
        {
            size_t end = index.back();
            for (size_t feature_idx=0; feature_idx<n_features; feature_idx++)
            {
                size_t start = index[feature_idx];
                new_index.push_back(current_col);
                if (interaction_only_)
                {
                    start += index[feature_idx+1] - index[feature_idx];
                }
                size_t next_col = current_col + end - start;
                if (next_col <= current_col)
                {
                    break;
                }

                auto next_feature = X[][feature_idx];
                for (auto&& [row, feature_row]: std::views::zip(XP, next_feature))
                {
                    auto row_begin = std::begin(row);                    
                    std::transform(row_begin+start, row_begin+end, row_begin+current_col, [feature_row](float a) { return a*feature_row; });
                }

                current_col = next_col;
            }
            new_index.push_back(current_col);
            // No copies, equivalent to auto v_t(std::move(new_index)); new_index = std::move(index); index = std::move(v_t);
            std::swap(index, new_index);
            new_index.clear();
        }
        if (min_degree > 1)
        {
            size_t n_XP = n_out_full_, n_Xout = n_features_out_;
            Array2D<float> Xout(n_samples, n_Xout);
            if (include_bias_)
            {
                ranges::fill(Xout[][0], 1);
                
                for (auto&& [row, row_out]: std::views::zip(XP, Xout))
                {
                    std::copy(std::begin(row)+n_XP - n_Xout + 1, std::end(row), std::begin(row_out)+1);
                }
            }
            else
            {
                for (auto&& [row, row_out]: std::views::zip(XP, Xout))
                {
                    std::copy(std::begin(row)+n_XP - n_Xout, std::end(row), std::begin(row_out));
                }
            }
            XP = std::move(Xout);
        }
        return XP;
    }
    //void fit_transform(Array2D<float>& X);

};
} // NAMESPACE ML