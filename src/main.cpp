#include <iostream>
#include <vector>
#include <random>
#include <format>
#include <algorithm>
#include <cmath>
#include <ranges>

template <typename T>
using Vector2D = std::vector<std::vector<T>>;

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
void print_line(const Vector2D<float>& v, int max)
{
    for (int i=0; i<max; i++)
    {
        for (int j=0; j<max; j++)
        {
            auto proj = [](auto& v){ return std::vector<int>{(int)std::round(v[0]), (int)std::round(v[1])}; };
            auto it = std::ranges::find(v, std::vector<int>{i, j}, proj);
            if (it == std::end(v))
            {
                std::cout << ' ';
            }
            else
            {
                std::cout << '*';
            }
        }
        std::cout << '\n';
    }
}
int main()
{
    auto v = gen_line_points(10, 2.1, 1.3);
    std::vector<float> X = std::move(v[0]);
    std::vector<float> y = std::move(v[1]);
    std::cout << "Hello world\n";
    for (auto [X_i, y_i]: std::views::zip(X, y))
    {
        std::cout << std::format("For X={:.2f}, Y={:.2f}\n", X_i, y_i);
    }

    return 0;
}