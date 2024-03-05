#pragma once
#include <vector>
#include <variant>
#include <array2D.hpp>
#include <transformermixin.hpp>
namespace ML
{
class PolynomialFeatures : public TransformerMixin<PolynomialFeatures>
{
private:
    std::pair<int, int> degree_ = DEFAULT_DEGREE;
    bool interaction_only_ = false;
    bool include_bias_ = false;
public:
    static constexpr std::pair<int, int> DEFAULT_DEGREE = {2, 2};
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
                degree_ = {d, d};
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
        (void)X;
        return *this;
    }
    void transform(Array2D<float>& X) const
    {
        (void)X;
    }
    //void fit_transform(Array2D<float>& X);

};
} // NAMESPACE ML