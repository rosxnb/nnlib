#ifndef NN_H_
#define NN_H_

#include <iostream>

class Mat
{
public:
    Mat(size_t rows, size_t cols);
    Mat(size_t rows, size_t cols, float x);
    Mat(size_t rows, size_t cols, float* data);
    Mat(const Mat& other);
    Mat& operator=(const Mat& other);
    ~Mat();

    void randomise();
    void fill(float x);
    void sigmoid() const;
    [[nodiscard]] Mat get_row(size_t row) const;

    void operator+=(const Mat& b) const;
    Mat operator+(const Mat& b) const;
    Mat operator*(const Mat& b) const;
    [[nodiscard]] float& at(size_t row, size_t col) const;

    float* begin();
    float* end();
    [[nodiscard]] float* begin() const;
    [[nodiscard]] float* end() const;
    [[nodiscard]] size_t row_count() const;
    [[nodiscard]] size_t col_count() const;

private:
    size_t m_rows;
    size_t m_cols;
    float* p_data;

private:
    float* operator[](size_t row) const;
    static float sigmoid_activation(float x) ;
    friend std::ostream& operator<<(std::ostream& out, const Mat& m);
};

float rand_float();

#endif // NN_H_
