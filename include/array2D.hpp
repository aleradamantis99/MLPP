#pragma once

#include <vector>
#include <iterator>
#include <iostream>
#include <ranges>
#include <format>
template <typename T>
class Array2D
{
public:
    using Vector = std::vector<T>;
private:
    template <typename P>
    class Row
    {
    public:
        Row(P* s, P* e):
            start_(s), end_(e) {}
        
        constexpr T& operator[](std::size_t j)
        {
            return *(start_+j);
        }
        constexpr const T& operator[](std::size_t j) const
        {
            return *(start_+j);
        }
        constexpr size_t size()
        {
            return end_-start_;
        }

        constexpr P* begin() { return start_; }
        constexpr const P* begin() const { return start_; }

        constexpr P* cbegin() { return start_; }
        constexpr const P* cbegin() const { return start_; }

        constexpr P* end() { return end_; }
        constexpr const P* end() const { return end_; }

        constexpr P* cend() { return end_; }
        constexpr const P* cend() const { return end_; }
    private:
        P* start_;
        P* end_;
    };
    constexpr std::size_t index_from_pos(std::size_t i, std::size_t j) const
    {
        return i*cols_+j;
    }
    std::size_t rows_=0, cols_=0;
    Vector v_;
public:
    constexpr Array2D() = default;
    constexpr Array2D(std::size_t rows, std::size_t cols, const T& value):
        rows_(rows),
        cols_(cols),
        v_(rows_*cols_, value) 
    {}

    constexpr Array2D(std::size_t rows, std::size_t cols):
        rows_(rows),
        cols_(cols),
        v_(rows_*cols_) 
    {}

    constexpr Array2D(std::initializer_list<std::initializer_list<T>> init):
        rows_(init.size()),
        cols_(init.begin()->size()),
        v_(rows_*cols_)
    {
        for (const auto& [i, d]: init | std::views::join | std::views::enumerate)
        {
            v_[i] = d;
        }
    }

    constexpr std::pair<std::size_t, std::size_t> shape() const
    {
        return {rows_, cols_};
    }

    constexpr T& at(std::size_t i, std::size_t j)
    {
        if (i>rows_ or j>cols_) throw std::out_of_range(std::format("Indexes out of bounds: {} and {}", rows_, cols_));
        return *(this)(i, j);
    }
    constexpr const T& at(std::size_t i, std::size_t j) const
    {
        if (i>rows_ or j>cols_) throw std::out_of_range(std::format("Indexes out of bounds: {} and {}", rows_, cols_));
        return *(this)(i, j);
    }

    constexpr T& operator()(std::size_t i, std::size_t j)
    {
        return v_[index_from_pos(i, j)];
    }
    constexpr const T& operator()(std::size_t i, std::size_t j) const
    {
        return v_[index_from_pos(i, j)];
    }

    constexpr auto operator[](std::size_t i)
    {
        return Row(v_.data()+i*cols_, v_.data()+i*cols_+rows_);
    }
    constexpr const auto operator[](std::size_t i) const
    {
        return Row(v_.data()+i*cols_, v_.data()+i*cols_+rows_);
    }
};


template <typename T>
std::ostream& operator<<(std::ostream& os, const Array2D<T>& a)
{
    for (std::size_t i=0; i<a.shape().first; i++)
    {
        for (std::size_t j=0; j<a.shape().second; j++)
        {
            os << a(i, j) << ',';
        }
        os << '\n';
    }
    return os;
}
