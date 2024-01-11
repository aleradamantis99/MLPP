#include <iostream>

int main()
{
    int data[][2] = {{0,1}, {2, 4}};
    std::cout << "Hello world\n";
    for (auto a: data)
    {
        for (int i=0; i<2; ++i)
        {
            std::cout << a[i] << ',';
        }
        std::cout << '\n';
    }
    return 0;
}