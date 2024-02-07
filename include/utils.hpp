#pragma once
#include <concepts>
#include <functional>

template <typename T>
concept TwoDimensionalAccesible = requires(T a)
{
    a[0ull][0ull];
    { a.size() } -> std::convertible_to<std::size_t>; 
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

//template <typename F>
//concept GradSigCallable = requires(F f, const std::vector<float>& X, const std::vector<float>& y, float w, float b) { f(X, y, w, b); };
