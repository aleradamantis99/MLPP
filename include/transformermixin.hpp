#pragma once
#include "array2D.hpp"

namespace ML
{
template <typename D>
class TransformerMixin
{
public:
    void fit_transform(Array2D<float>& X)
    {
        this->underlying().fit(X);
        this->underlying().transform(X);
    }
};
} //namespace ML