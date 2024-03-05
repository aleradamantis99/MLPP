#pragma once
#include "array2D.hpp"
#include "crtp.hpp"
namespace ML
{
template <typename D>
class TransformerMixin: public CRTP<D, TransformerMixin>
{
public:
    void fit_transform(Array2D<float>& X)
    {
        this->underlying().fit(X);
        this->underlying().transform(X);
    }
};
} //namespace ML