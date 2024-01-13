#include <iostream>
#include <functional>
#include <format>

template <typename F>
auto do_twice_template(F f1, F f2)
{
    return f1(f2(3));
}

template <typename F1, typename F2>
auto do_twice_template2types(F1 f1, F2 f2)
{
    return f1(f2(3));
}

using int_func = int(*)(int);
auto do_twice_pointer(int_func f1, int_func f2)
{
    return f1(f2(3));
}


auto do_twice_function(std::function<int(int)> f1, std::function<int(int)> f2)
{
    return f1(f2(3));
}

template <typename F, typename CallableSignature>
concept Callable = std::is_convertible_v<F, std::function<CallableSignature>>;

template <Callable<int(int)> F>
auto do_twice_concept1(F f1, F f2)
{
    return f1(f2(3));
}

auto do_twice_auto(auto f1, auto f2)
{
    return f1(f2(3));
}

auto do_twice_concept2(Callable<int(int)> auto f1, Callable<int(int)> auto f2)
{
    return f1(f2(3));
}



int square(int x)
{
    return x*x;
}

int triple(int x)
{
    return 3*x;
}

struct Square
{
    int operator()(int x)
    {
        return x*x;
    }
} square_o;
struct Triple
{
    int num = 3;
    int operator()(int x)
    {
        return num*x;
    }
} triple_o;
int main()
{
    auto l_square = [] (int x) { return x*x; };
    auto l_triple = [] (int x) { return 3*x; };

    std::cout << std::format("___NORMAL FUNCTION___\nTemplate={}\nTemplate With Two Types={}\nPointer={}\nstd::function={}\nConcept1={}\nConcept2={}\n", do_twice_template(square, triple), do_twice_template2types(square, triple), do_twice_pointer(square, triple), do_twice_function(square, triple), do_twice_concept1(square, triple), do_twice_concept2(square, triple));
    std::cout << std::format("___LAMBDA FUNCTION___\nTemplate=NO FUNCIONA\nTemplate With Two Types={}\nPointer={}\nstd::function={}\nConcept1=NO FUNCIONA\nConcept2={}\n", do_twice_template2types(l_square, l_triple), do_twice_pointer(l_square, l_triple), do_twice_function(l_square, l_triple), do_twice_concept2(l_square, l_triple));
    std::cout << std::format("___LAMBDA FUNCTION___\nTemplate=NO FUNCIONA\nTemplate With Two Types={}\nPointer=NO FUNCIONA\nstd::function={}\nConcept1=NO FUNCIONA\nConcept2={}\n", do_twice_template2types(square_o, triple_o), do_twice_function(square_o, triple_o), do_twice_concept2(square_o, triple_o));

    return 0;
}