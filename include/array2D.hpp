#pragma once

#include <vector>
#include <iterator>
#include <iostream>
#include <ranges>
#include <format>
#include <cassert>
#include <span>
namespace ML
{
template <typename T>
class Array2D
{
    template <typename P>
    using Row_ = std::span<P, std::dynamic_extent>;

    template <typename P>
    struct Columns_;

    using ConstColumns = Columns_<const T>;
    using Columns = Columns_<T>;
public:
    using ConstColumn = ConstColumns::Column;
    using Column = Columns::Column;
    using Vector = std::vector<T>;
    

    template <typename P>
    struct Iterator
    {
        using iterator_category = std::random_access_iterator_tag;
        using value_type = std::span<P>;
        using difference_type = std::ptrdiff_t;
        using pointer = std::span<P>*;
        using reference = std::span<P>&&;
        constexpr Iterator() = default;
        constexpr Iterator(P* start, size_t cols):
            start_(start), cols_(cols)
            {}
        constexpr value_type operator*() const
        {
            return value_type(start_, start_+cols_);
        }
        constexpr Iterator& operator++()
        {
            start_ += cols_;
            return *this;
        }
        constexpr Iterator operator++(int)
        {
            auto oldthis = *this;
            start_ += cols_;
            return oldthis;
        }
        constexpr Iterator& operator--()
        {
            start_ -= cols_;
            return *this;
        }
        constexpr Iterator operator--(int)
        {
            auto oldthis = *this;
            start_ -= cols_;
            return oldthis;
        }
        constexpr Iterator& operator+=(size_t offset)
        {
            start_ += offset*cols_;
            return *this;
        }
        constexpr Iterator& operator-=(size_t offset)
        {
            return operator+=(-offset);
        }
        constexpr Iterator operator+(size_t offset) const
        {
            Iterator new_it = *this;
            return new_it += offset;
        }
        constexpr Iterator operator-(size_t offset) const
        {
            Iterator new_it = *this;
            return new_it -= offset;
        }
        constexpr std::ptrdiff_t operator-(const Iterator& it) const
        {
            return start_-it.start_;
        }
        friend constexpr Iterator operator+(size_t offset, const Iterator& it)
        {
            return it+offset;
        }

        constexpr auto operator<=>(const Iterator& rhs) const = default;

        constexpr value_type operator->()
        {
            return std::span<P>(start_, cols_);
        }

        constexpr value_type operator[](size_t offset) const
        {
            return std::span<P>(start_+offset*cols_, cols_);
        }
    private:
        P* start_=nullptr;
        size_t cols_=0;
    };
    constexpr std::size_t index_from_pos(std::size_t i, std::size_t j) const
    {
        return i*cols_+j;
    }
    std::size_t rows_=0, cols_=0;
    Vector v_;
public:
    using iterator = Iterator<T>;
    using const_iterator = Iterator<const T>;
    using Row = std::span<T>;
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
    {
        //auto view = std::views::chunk(v_, cols);
    }

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

    constexpr Array2D(std::vector<T> v, size_t rows, size_t cols):
        rows_(rows),
        cols_(cols),
        v_(std::move(v)) 
    {}

    constexpr size_t size() const
    {
        return rows_;
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
        return std::span(v_.data()+i*cols_, cols_);
    }
    constexpr const auto operator[](std::size_t i) const
    {
        return std::span(v_.data()+i*cols_, cols_);
    }

    constexpr auto operator[]()
    {
        return Columns(cols_, rows_, v_.data());
    }
    constexpr const auto operator[]() const
    {
        return ConstColumns(cols_, rows_, v_.data());
    }

    constexpr iterator begin() 
    {
        return {v_.data(), cols_}; 
    }
    constexpr const_iterator begin() const { return {v_.data(), cols_}; }

    constexpr const_iterator cbegin() const { return begin(); }

    constexpr iterator end() { return {v_.data()+cols_*rows_, cols_}; }
    constexpr const_iterator end() const { return {v_.data()+cols_*rows_, cols_}; }

    constexpr const_iterator cend() const { return end(); }

};

template <typename T>
template <typename P>
struct Array2D<T>::Columns_
{
    struct Column
    {
        using ViewType = decltype(std::ranges::subrange(std::declval<P*>(), std::declval<P*>()) | std::views::stride(std::declval<size_t>()));

        ViewType column_;
        size_t size_;
        constexpr Column(P* begin, P* end, size_t cols, size_t rows):
            column_(std::ranges::subrange(begin, end) | std::views::stride(cols)),
            size_(rows)
        {}

        constexpr size_t size() const
        {
            return size_; //Could also do return column_.size() so we don't have to store size_, but that involves some calculations.
        }

        constexpr auto begin()
        {
            return std::begin(column_);
        }

        constexpr auto begin() const { return std::cbegin(column_); }

        constexpr auto cbegin() const { return begin(); }

        constexpr auto end() { return std::end(column_); }
        constexpr auto end() const { return std::cend(column_); }

        constexpr auto cend() const { return end(); }
    };
    constexpr Column operator[](size_t i)
    {
        auto start = begin+i;
        auto end = start + cols_*(rows_-1)+1;
        assert(start < end and end <= begin+rows_*cols_);
        return Column(start, start + cols_*(rows_-1)+1, cols_, rows_);
    }
    constexpr Column operator[](size_t i) const
    {
        auto start = begin+i;
        auto end = start + cols_*(rows_-1)+1;
        assert(start < end and end <= begin+rows_*cols_);
        return Column(start, start + cols_*(rows_-1)+1, cols_, rows_);
    }
    size_t cols_, rows_;
    P* begin;
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


}