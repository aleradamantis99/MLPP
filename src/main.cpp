#include <iostream>
#include <vector>
#include <random>
#include <format>
#include <algorithm>
#include <cmath>
#include <ranges>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>


#include <array2D.hpp>
#include <zscorenormalizer.hpp>
#include <utils.hpp>
#include <linearregression.hpp>
#include <logsticregression.hpp>
#include <mlcommons.hpp>
#include <polynomialfeatures.hpp>
namespace ranges = std::ranges;
using namespace ML;

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

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ','))
    {
        if (cell[cell.size()-1] == '\r')
        {
            cell.erase(cell.size()-1);
        }
        result.push_back(cell);
    }

    
    return result;
}
auto read_csv(const std::string& filename)
{
    std::ifstream csv_file(filename);
    size_t n_columns = getNextLineAndSplitIntoTokens(csv_file).size()-1;
    std::vector<float> X;
    std::vector<int> y;
    size_t rows = 0;
    while(csv_file)
    {
        std::vector<std::string> line = getNextLineAndSplitIntoTokens(csv_file);
        if(line.empty()) continue;

        for (const auto& f: line | ranges::views::take(n_columns))
        {
            float value = std::stof(f);
            X.push_back(value);
        }

        try
        {
            y.push_back(std::stoi(line.at(n_columns)));
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
        
        rows++;
    }
    X.shrink_to_fit();
    y.shrink_to_fit();
    Array2D a_X (std::move(X), rows, n_columns);
    return std::pair<Array2D<float>, std::vector<int>>{std::move(a_X), std::move(y)};
}

void println(auto v)
{
    std::cout << v << std::endl;
}

template <typename>
struct TD;


int main()
{
    [[maybe_unused]] PolynomialFeatures pf({.degree = std::pair{2, 3}, .interaction_only=true});
    std::cout << is_classifier<LinearRegression>() << std::endl;
    /*auto cancer = read_csv("haberman.csv");

    Array2D<float>& X = cancer.first;
    std::vector<int>& y = cancer.second;
    ZScoreNormalizer norm;
    norm.fit_transform(X);

    LogisticRegression lr({.learning_rate=0.1, .max_iter=1000});
    
    lr.fit(X, y);
    std::cout << lr.score(X, y);
    */

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
    std::cout << std::format("R2 for base dataset: {}\nR2 for test: {}\n", R2, R2_test);
    //std::cout << std::format("Found w1:{} w2:{} and b:{} through gradient descent (Cost: {})\n", gd_w[0], gd_w[1], gd_b, cost);
}