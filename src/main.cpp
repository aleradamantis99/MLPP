#include <iostream>
#include <vector>
#include <random>
#include <format>
#include <algorithm>
#include <cmath>
#include <ranges>
#include <functional>
#include <numeric>
#include <fstream>

#include <array2D.hpp>
#include <zscorenormalizer.hpp>

namespace ranges = std::ranges;

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

Vector2D<float> gen_line_points(size_t n_points, float w, float b, float noise = 0.f)
{
    Vector2D<float> v (2, std::vector<float>(n_points));
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.f,20.f); 
    std::uniform_real_distribution<float> noise_dist(-noise,noise); 
    for (auto [x, y]: std::views::zip(v[0], v[1]))
    {
        x = dist(rng);
        y = (x*w + b) + noise_dist(rng);
    }
    return v;
}

float cost_function(const std::vector<float>& X, const std::vector<float>& y, float w, float b)
{
    auto n = X.size();
    float total_cost = 0.f;
    for (auto [X_i, y_i]: std::views::zip(X, y))
    {
        float y_pred = w*X_i + b;
        float diff = (y_i - y_pred)*(y_i - y_pred);
        total_cost += diff;
    }
    return total_cost/(2*n);
}

float dot_product(const std::vector<float>& a, const std::vector<float>& b)
{
    return std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.f);
}

std::pair<std::vector<float>, float> cost_gradient(const Vector2D<float>& X, const std::vector<float>& y, const std::vector<float>& w, float b)
{
    size_t n = X.size();    
    std::vector<float> dj_dw(X[0].size(), 0);
    float dj_db = 0;

    for (size_t i=0; i<n; i++)
    {
        float f_wb = dot_product(w, X[i]) + b;
        float err = f_wb - y[i];
        ranges::transform(X[i], dj_dw, std::begin(dj_dw), [err, n](float x, float dw) { return err*x + dw; });
        dj_db += err;
    }
    ranges::transform(dj_dw, std::begin(dj_dw), [&n](float dw) { return dw/n; });
    return {std::move(dj_dw), dj_db/n};
}

template <GradSigCallable Gf>
std::pair<std::vector<float>, float> gradient_descent(const Vector2D<float>& X, const std::vector<float>& y, float alpha, size_t num_iters, Gf gradient_function)
{
    float b = 0;
    std::vector<float> w(X[0].size(), 0);
    
    for (size_t i=0; i<num_iters; ++i)
    {
        auto [dj_dw, dj_db] = gradient_function(X, y, w, b);
        b = b - alpha*dj_db;

        ranges::transform(w, dj_dw, std::begin(w), [alpha](float w,  float dw) { return w-alpha*dw; });
    }

    return {std::move(w), b};
}


std::pair<Vector2D<float>, std::vector<float>> gen_house_prices(size_t n_houses, float noise = 0, std::size_t seed = -1)
{
    if (seed == static_cast<std::size_t>(-1))
    {
        std::random_device dev;
        seed = dev();
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> size_dist(30.f,200.f); 
    std::uniform_real_distribution<float> noise_dist(-noise,noise); 
    std::uniform_int_distribution<int> room_dist(0,5); 
    std::vector<float> house_size(n_houses);
    ranges::generate(house_size, [&dist = size_dist, &rng] { return dist(rng); });
    std::vector<float> rooms(n_houses);
    ranges::generate(rooms, [&dist = room_dist, &rng] { return static_cast<float>(dist(rng)); });
    std::vector<float> price (n_houses);
    ranges::transform(house_size, rooms, std::begin(price), [&noise_dist, &rng](float hs, float r) { return (10.f+0.1f*hs+r)+noise_dist(rng); });

    return {Vector2D<float>{std::move(house_size), std::move(rooms)}, std::move(price)};
}

void write_to_csv(const std::string& filename, const Vector2D<float>& X, const std::vector<float>& y)
{
    std::ofstream houses_file(filename);
    std::ostream_iterator<char> out(houses_file);
    std::format_to(out, "precio,metros_cuadrados,habitaciones\n");
    for (size_t i=0; i<y.size(); ++i)
    {
        std::format_to(out, "{},{},{}\n", y[i], X[0][i], X[1][i]);
    }
}
template <typename>
struct TD;

int main()
{
    Array2D<float> a({{1, 2}, {3, 4}});
    std::cout << a << '\n';
    [](const auto& a){ std::cout << a[1][0] <<'\n'; }(a);

    /*auto houses = gen_house_prices(100, 0, 422);
    Vector2D<float>& X = houses.first;
    std::vector<float>& y = houses.second;

    write_to_csv("houses.csv", X, y);

    ZScoreNormalizer norm;
    norm.fit_transform(X);

    
    write_to_csv("houses_norm.csv", X, y);
    auto [gd_w, gd_b] = gradient_descent(X, y, 0.0001, 100000, cost_gradient);
    std::cout << std::format("Found w1:{} w2:{} and b:{} through gradient descent\n", gd_w[0], gd_w[1], gd_b);*/

    return 0;
}