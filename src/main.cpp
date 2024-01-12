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

std::pair<float, float> cost_gradient(const std::vector<float>& X, const std::vector<float>& y, float w, float b)
{
    size_t n = X.size();    
    float dj_dw = 0;
    float dj_db = 0;
    
    for (size_t i=0; i<n; i++)
    {
        float f_wb = w*X[i] + b;
        dj_dw += (f_wb - y[i]) * X[i];
        dj_db += f_wb - y[i];
    }
        
    return {dj_dw/n, dj_db/n};
}

std::pair<float, float> gradient_descent(const std::vector<float>& X, const std::vector<float>& y, float w0, float b0, float alpha, size_t num_iters, cost_functions, gradient_function)
{

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
    std::cout << cost_function(X, y, 2.1, 0) << '\n';
    float w=2.1, b=1.1;
    auto [dj_dw, dj_db] = cost_gradient(X, y, w, b);
    std::cout << std::format("Gradient at w={} and b={}: w: {}, b: {}\n", w, b, dj_dw, dj_db);
    return 0;
}