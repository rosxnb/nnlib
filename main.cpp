#include "nn.hpp"

int main()
{
    std::cout.setf(std::ios::fixed);

    Mat m(2, 2);
    Mat n(2, 2);

    m.fill(1.f);
    n.fill(1.f);
    std::cout << m << std::endl;
    std::cout << n;

    m += n;
    std::cout << "---------------------------------------------------------------------------\n";
    std::cout << m << std::endl;
    std::cout << "---------------------------------------------------------------------------\n";

    Mat a(4, 4, 1.f);
    std::cout << a << std::endl;

    Mat b(4, 2);
    b.randomise();
    std::cout << b;

    std::cout << "---------------------------------------------------------------------------\n";
    Mat res = a * b;
    std::cout << res;
    std::cout << "---------------------------------------------------------------------------\n";

    return 0;
}
