#include <iostream>
#include <vector>
#include <random>
#include <format>
#include <algorithm>
#include <cmath>
#include <ranges>
#include <numeric>
#include <fstream>

#include <array2D.hpp>
#include <zscorenormalizer.hpp>
#include <utils.hpp>
#include <linearregression.hpp>
#include <mlcommons.hpp>
namespace ranges = std::ranges;
using namespace ML;
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


std::pair<Array2D<float>, std::vector<float>> gen_house_prices(size_t n_houses, float noise = 0, std::size_t seed = -1)
{
    if (seed == static_cast<std::size_t>(-1))
    {
        std::random_device dev;
        seed = dev();
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> size_dist(30.f,200.f); 
    std::normal_distribution<float> noise_dist(0,noise); 
    std::normal_distribution<float> room_dist(2.5,1.5);
    auto gen_size = [&dist = size_dist, &rng] { return dist(rng); };
    auto gen_rooms = [&dist = room_dist, &rng] { auto r = 0.f+std::ceil(dist(rng)); return r >= 0.f? +r : std::ceil(dist.mean()); };
    auto calc_price = [&noise_dist, &rng](float hs, float r) { return (10.f+0.1f*hs+r)+noise_dist(rng); };

    Array2D<float> house_data(n_houses, 2);
    std::vector<float> price(n_houses);
    for (std::size_t i=0; i<n_houses; i++)
    {
        auto house = house_data[i];
        house[0] = gen_size();
        house[1] = gen_rooms();
        price[i] = calc_price(house[0], house[1]);
    }
    /*std::vector<float> house_size(n_houses);
    ranges::generate(house_size, [&dist = size_dist, &rng] { return dist(rng); });
    std::vector<float> rooms(n_houses);
    ranges::generate(rooms, [&dist = room_dist, &rng] { return static_cast<float>(dist(rng)); });
    std::vector<float> price (n_houses);
    ranges::transform(house_size, rooms, std::begin(price), [&noise_dist, &rng](float hs, float r) { return (10.f+0.1f*hs+r)+noise_dist(rng); });*/

    return {std::move(house_data), std::move(price)};
}
template <TwoDimensionalAccesible T>
void write_to_csv(const std::string& filename, const T& X, const std::vector<float>& y)
{
    std::ofstream houses_file(filename);
    std::ostream_iterator<char> out(houses_file);
    std::format_to(out, "precio,metros_cuadrados,habitaciones\n");
    for (size_t i=0; i<y.size(); ++i)
    {
        std::format_to(out, "{},{},{}\n", y[i], X[i][0], X[i][1]);
    }
}
template <typename>
struct TD;

int main()
{
    Array2D<float> a({{1, 2}, {3, 4}});
    std::cout << a << '\n';
    [](const auto& a){ std::cout << a[1][0] <<'\n'; }(a);

    auto houses = gen_house_prices(100, 0, 422);
    Array2D<float>& X = houses.first;
    std::vector<float>& y = houses.second;

    auto houses_test = gen_house_prices(20, 0, 123);
    Array2D<float>& X_test = houses_test.first;
    std::vector<float>& y_test = houses_test.second;

    write_to_csv("houses.csv", X, y);

    for (auto [h, p]: std::views::zip(X, y))
    {
        for (auto f: h)
        {
            std::cout << f << ',';
        }
        std::cout << p << '\n';
    }
    ZScoreNormalizer norm;
    norm.fit_transform(X);
    norm.transform(X_test);
    write_to_csv("houses_norm.csv", X, y);
    LinearRegression lr(0.1, 1000);

    lr.fit(X, y);
    std::vector<float> y_pred = lr.predict(X);

    float R2 = lr.score(X, y);
    float R2_test = lr.score(X_test, y_test);
    std::cout << std::format("R2 for base dataset: {}\nR2 for test: {}", R2, R2_test);
    //std::cout << std::format("Found w1:{} w2:{} and b:{} through gradient descent (Cost: {})\n", gd_w[0], gd_w[1], gd_b, cost);
}