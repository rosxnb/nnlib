#ifndef NN_H_
#define NN_H_

#include <iostream>

class Mat
{
public:
    void print(size_t i, size_t j)
    {
        std::cout << ( (p_data + i * m_cols) [j] ) << std::endl;
    }

public:
    Mat(size_t rows, size_t cols);
    Mat(size_t rows, size_t cols, float x);
    ~Mat();

    void randomise(void);
    void fill(float x);

    float* begin();
    float* end();

    void operator+=(const Mat& b);
    Mat operator*(const Mat& b);
    float* operator[](size_t row) const;
    friend std::ostream& operator<<(std::ostream& out, const Mat& m);

private:
    size_t m_rows;
    size_t m_cols;
    float* p_data;
};

float rand_float();

#endif // NN_H_
