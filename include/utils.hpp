#pragma once
#include <concepts>
#include <ranges>
#include <functional>
#include <numeric>
#include <vector>

#include <generator.hpp>
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


constexpr auto binomial_coefficient(std::integral auto n, std::integral auto k) {
    using type = decltype(k);
    std::vector<type> aSolutions(k);
    aSolutions[0] = n - k + 1;

    for (type i = 1; i < k; ++i) {
    aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
    }

    return aSolutions[k - 1];
}
template <std::ranges::random_access_range I>
coro::generator<coro::generator<unsigned>> combinations_with_replacement(I iterable, unsigned r)
{
    namespace ranges = std::ranges;
    size_t n = iterable.size();
    if (not n and r) co_return;
    std::vector indices(r, 0u);
    auto gen_in = [&iterable, &indices]() -> coro::generator<unsigned> { for (unsigned i: indices) co_yield iterable[i]; };
    co_yield gen_in();
    for(;;)
    {
        unsigned i;
        for (i=r-1; i!=(unsigned)-1 and indices[i] == n - 1; i--) { }
        if (i==(unsigned)-1) co_return;
        std::fill(std::begin(indices)+i, std::end(indices), indices[i] + 1);
        //std::fill_n(std::begin(indices)+i, r-i, indices[i]+1);
        co_yield gen_in();
    }
}


template <typename T>
concept Printable = requires(T a)
{
    std::cout << a << std::endl;
};
template <typename T>
concept ElementPrintable = not Printable<T> and requires(T a)
{
    std::cout << *std::begin(a) << std::endl;
};
template <typename T>
concept SubElementPrintable = not ElementPrintable<T> and not Printable<T> and requires(T a)
{
    std::cout << *std::begin(*std::begin(a)) << std::endl;
};
void print(SubElementPrintable auto&& v, char delimiter = ' ', char outer_delimiter = '\n')
{
    for (auto&& e: v)
    {
        for (auto&& sub_e: e)
        {
            std::cout << sub_e << delimiter;
        }
        std::cout << outer_delimiter;
    }
}
void print(const ElementPrintable auto& v, char delimiter = '\n')
{
    for (const auto& e: v)
    {
        std::cout << e << delimiter;
    }
}
void print(Printable auto&& v, char delimiter = '\n') 
{
    std::cout << v << delimiter;
}

template <std::ranges::random_access_range I>
coro::generator<coro::generator<unsigned>> combinations(I iterable, unsigned r)
{
    namespace ranges = std::ranges;
    size_t n = iterable.size();

    if (r > n) co_return;

    std::vector<unsigned> indices(r);
    ranges::iota(indices, 0);

    /*std::vector<unsigned> result_(r);
    [[maybe_unused]] auto modify = [&] { 
        for (size_t i=0; i<r; i++) 
            result_[i] = iterable[indices[i]]; 
    };
    modify();
    co_yield result_;
    */
    auto gen_in = [&iterable, &indices]() -> coro::generator<unsigned> { for (unsigned i: indices) co_yield iterable[i]; };
    co_yield gen_in();
    for(;;)
    {
        unsigned i;
        /*bool none = ranges::none_of(std::views::iota(0u, r) | std::views::reverse, [&] (unsigned inx)
        {
            i = inx;
            return indices[i] != i + n - r;
        });
        if none co_return;*/
        
        for (i=r-1; i!=(unsigned)-1 and indices[i] == i + n - r; i--) { }
        if (i==(unsigned)-1) co_return;

        indices[i] += 1;
        for (auto j: std::ranges::iota_view(i+1, r))
        {
            indices[j] = indices[j-1] + 1;
        }
        co_yield gen_in();
    }
}
//template <typename F>
//concept GradSigCallable = requires(F f, const std::vector<float>& X, const std::vector<float>& y, float w, float b) { f(X, y, w, b); };
}