#include "nn.hpp"

#include <algorithm>
#include <chrono>
#include <random>

Mat::Mat(size_t rows, size_t cols)
    : m_rows(rows), m_cols(cols)
{
    p_data = new float[m_rows * m_cols];
    assert(p_data != nullptr);
    fill(0.f);
}

Mat::Mat(size_t rows, size_t cols, float x)
    : m_rows(rows), m_cols(cols)
{
    assert(rows == cols);

    p_data = new float[m_rows * m_cols];
    assert(p_data != nullptr);

    for (size_t i = 0; i < m_rows; ++i) {
        (*this)[i][i] = x;
    }
}

Mat::~Mat()
{
    delete[] p_data;
}

void Mat::operator+=(const Mat& b)
{
    assert(m_rows == b.m_rows);
    assert(m_cols == b.m_cols);

    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            (*this)[i][j] += b[i][j];
        }
    }

}

Mat Mat::operator*(const Mat& b)
{
    assert(m_cols == b.m_rows);

    Mat res(m_rows, b.m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < b.m_cols; ++j) {
            for (size_t k = 0; k < m_cols; ++k) {
                res[i][j] += (*this)[i][k] * b[k][j];
            }
        }
    }
    return res;
}

void Mat::randomise(void)
{
    std::generate(begin(), end(), rand_float);
}

void Mat::fill(float x)
{
    std::fill(begin(), end(), x);
}

float* Mat::begin()
{
    return p_data;
}

float* Mat::end()
{
    return p_data + m_rows * m_cols;
}

float* Mat::operator[](size_t row) const
{
    return p_data + (row * m_cols);
}

std::ostream& operator<<(std::ostream& out, const Mat& mat)
{
    out << "[\n";
    for (size_t i = 0; i < mat.m_rows; ++i) {
        out << "    ";
        for (size_t j = 0; j < mat.m_cols; ++j) {
            out << mat[i][j] << ' ';
        }
        out << '\n';
    }
    out << "]\n";

    return out;
}

float rand_float()
{
    static auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    static std::mt19937 eng(seed);
    static std::uniform_real_distribution<float> gen(0, 1);
    return gen(eng);
}
