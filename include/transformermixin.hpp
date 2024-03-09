#pragma once
#include "array2D.hpp"
#include "crtp.hpp"
namespace ML
{
template <typename D>
class TransformerMixin: public CRTP<D, TransformerMixin>
{
public:
    [[nodiscard]] Array2D<float> fit_transform(const Array2D<float>& X)
    {
        this->underlying().fit(X);
        return this->underlying().transform(X);
    }
};
} //namespace ML