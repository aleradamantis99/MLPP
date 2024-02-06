#pragma once

template <typename T>
concept TwoDimensionalAccesible = requires(T a)
{
    a[0ull][0ull];
    
};