#pragma once
#include <concepts>
#include <ranges>
#include <functional>
#include <numeric>

namespace ML
{
namespace ranges = std::ranges;
template <typename T>
concept OneDimensionalAccesible = std::ranges::random_access_range<T> and std::ranges::sized_range<T>;

template <typename T>
concept TwoDimensionalAccesible = OneDimensionalAccesible<T> and requires(T a)
{
    a[0ull][0ull];
    { a[0].size() } -> std::convertible_to<std::size_t>; 
};

template <typename T>
using Vector2D = std::vector<std::vector<T>>;

template <typename F, typename CallableSignature>
concept Callable = std::is_convertible_v<F, std::function<CallableSignature>>;

template <typename F>
concept CostSigCallable = Callable<F, float(const std::vector<float>&, const std::vector<float>&, float, float)>;

template <typename F>
concept GradSigCallable = Callable<F, std::pair<std::vector<float>, float>(const Vector2D<float>&, const std::vector<float>&, const std::vector<float>&, float)>;

template <typename F>
concept GradSigCallableSTD = std::regular_invocable<F, std::pair<std::vector<float>, float>(const Vector2D<float>&, const std::vector<float>&, const std::vector<float>&, float)>;

float dot_product(const ranges::range auto& a, const ranges::range auto& b)
{
    return std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.f);
}

//template <typename F>
//concept GradSigCallable = requires(F f, const std::vector<float>& X, const std::vector<float>& y, float w, float b) { f(X, y, w, b); };
}