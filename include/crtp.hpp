#pragma once

template <typename T, template<typename> class crtpType>
class CRTP
{
    T& underlying() { return static_cast<T&>(*this); }
    T const& underlying() const { return static_cast<T const&>(*this); }
    CRTP()
    {
        static_assert(std::derived_from<T, crtpType<T>>, "Only use CRTP with same class as derived: class Derived : Mixin<Derived>.");
    }
    friend crtpType<T>;
};