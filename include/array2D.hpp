#pragma once

#include <vector>
#include <iterator>
#include <iostream>
#include <ranges>
#include <format>
#include <cassert>
template <typename T>
class Array2D
{
public:
    using Vector = std::vector<T>;
    template <typename P>
    class Row
    {
    public:
        Row(P* s, P* e):
            start_(s), end_(e) {}
        
        constexpr T& at(std::size_t j)
        {
            auto it = *(start_+j)
            assert(it < end_);
            return *(start_+j);
        }
        constexpr const T& at(std::size_t j) const
        {
            auto it = *(start_+j)
            assert(it < end_);
            return *(start_+j);
        }

        constexpr T& operator[](std::size_t j)
        {
            return *(start_+j);
        }
        constexpr const T& operator[](std::size_t j) const
        {
            return *(start_+j);
        }
        constexpr size_t size() const
        {
            return end_-start_;
        }

        constexpr P* begin() { return start_; }
        constexpr const P* begin() const { return start_; }

        constexpr const P* cbegin() const { return start_; }

        constexpr P* end() { return end_; }
        constexpr const P* end() const { return end_; }

        constexpr const P* cend() const { return end_; }
    private:
        P* start_;
        P* end_;
    };
private:    
    template <typename P>
    struct Iterator
    {
        using iterator_category = std::random_access_iterator_tag;
        using value_type = Row<P>;
        using difference_type = std::ptrdiff_t;
        using pointer = Row<P>*;
        using reference = Row<P>&&;
        constexpr Iterator() = default;
        constexpr Iterator(P* start, size_t cols):
            start_(start), cols_(cols)
            {}
        constexpr Row<P> operator*() const
        {
            return Row(start_, start_+cols_);
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
        constexpr size_t operator-(const Iterator& it) const
        {
            return start_-it.start_;
        }
        friend constexpr Iterator operator+(size_t offset, const Iterator& it)
        {
            return it+offset;
        }

        constexpr auto operator<=>(const Iterator& rhs) const = default;

        constexpr Row<P> operator->()
        {
            return Row(start_, start_+cols_);
        }

        constexpr Row<P> operator[](size_t offset) const
        {
            return Row(start_, start_+offset*cols_);
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
        return Row(v_.data()+i*cols_, v_.data()+i*cols_+rows_);
    }
    constexpr const auto operator[](std::size_t i) const
    {
        return Row(v_.data()+i*cols_, v_.data()+i*cols_+rows_);
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
