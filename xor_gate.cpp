#include "nn.hpp"

struct Xor
{
    Mat a0;
    Mat a1, w1, b1;
    Mat a2, w2, b2;

    Xor()
        : a0(1, 2), a1(1, 2), w1(2, 2), b1(1, 2), a2(1, 1), w2(2, 1), b2(1, 1)
    {
    }

    void randomize()
    {
        w1.randomise();
        b1.randomise();
        w2.randomise();
        b2.randomise();

    }
};

void forward_xor(Xor& m)
{
    m.a1 = m.a0 * m.w1 + m.b1;
    // m.a1 = m.a0 * m.w1;
    // m.a1 += m.b1;
    m.a1.sigmoid();

    m.a2 = m.a1 * m.w2 + m.b2;
    // m.a2 = m.a1 * m.w2;
    // m.a2 += m.b2;
    m.a2.sigmoid();
}

float cost(Xor m, Mat in, Mat out)
{
    assert(in.row_count() == out.row_count());
    assert(out.col_count() == m.a2.col_count());

    size_t x = in.row_count();
    float c = 0.f;

    for (size_t i = 0; i < x; ++i) {
        Mat a = in.get_row(i);
        Mat b = out.get_row(i);

        m.a0 = a;
        forward_xor(m);

        size_t y = out.col_count();
        for (size_t j = 0; j < y; ++j) {
            float d = m.a2.at(0, j) - b.at(0, j);
            c += d * d;
        }
    }

    return c / x;
}

void finite_diff(const Xor& m, Xor& g, float eps, const Mat& ti, const Mat& to)
{
    float saved;
    float c = cost(m, ti, to);

    for (size_t i = 0; i < m.w1.row_count(); ++i) {
        for (size_t j = 0; j < m.w1.col_count(); ++j) {
            saved = m.w1.at(i, j);
            m.w1.at(i, j) += eps;
            g.w1.at(i, j) = (cost(m, ti, to) - c) / eps;
            m.w1.at(i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b1.row_count(); ++i) {
        for (size_t j = 0; j < m.b1.col_count(); ++j) {
            saved = m.b1.at(i, j);
            m.b1.at(i, j) += eps;
            g.b1.at(i, j) = (cost(m, ti, to) - c) / eps;
            m.b1.at(i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.w2.row_count(); ++i) {
        for (size_t j = 0; j < m.w2.col_count(); ++j) {
            saved = m.w2.at(i, j);
            m.w2.at(i, j) += eps;
            g.w2.at(i, j) = (cost(m, ti, to) - c) / eps;
            m.w2.at(i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b2.row_count(); ++i) {
        for (size_t j = 0; j < m.b2.col_count(); ++j) {
            saved = m.b2.at(i, j);
            m.b2.at(i, j) += eps;
            g.b2.at(i, j) = (cost(m, ti, to) - c) / eps;
            m.b2.at(i, j) = saved;
        }
    }
}

void learn(const Xor& m, const Xor& g, float rate)
{
    for (size_t i = 0; i < m.w1.row_count(); ++i) {
        for (size_t j = 0; j < m.w1.col_count(); ++j) {
            m.w1.at(i, j) -= rate * g.w1.at(i, j);
        }
    }

    for (size_t i = 0; i < m.b1.row_count(); ++i) {
        for (size_t j = 0; j < m.b1.col_count(); ++j) {
            m.b1.at(i, j) -= rate * g.b1.at(i, j);
        }
    }

    for (size_t i = 0; i < m.w2.row_count(); ++i) {
        for (size_t j = 0; j < m.w2.col_count(); ++j) {
            m.w2.at(i, j) -= rate * g.w2.at(i, j);
        }
    }

    for (size_t i = 0; i < m.b2.row_count(); ++i) {
        for (size_t j = 0; j < m.b2.col_count(); ++j) {
            m.b2.at(i, j) -= rate * g.b2.at(i, j);
        }
    }
}

int main()
{
    std::cout.setf(std::ios::fixed);

    float ti[] {
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    };
    Mat in_samples(4, 2, ti);

    float to[] {
        0, 1, 1, 0,
    };
    Mat out_samples(4, 1, to);

    Xor m; m.randomize();
    Xor g;
    float eps = 1e-1;
    float lr  = 1e-1;


    size_t epochs  = 100000;
    while (epochs--) {
        finite_diff(m, g, eps, in_samples, out_samples);
        learn(m, g, lr);
        // std::cout << cost(m, in_samples, out_samples) << std::endl;
    }
    std::cout << cost(m, in_samples, out_samples) << "\n\n";

    #if 1
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            m.a0.at(0, 1) = (float)i;
            m.a0.at(0, 0) = (float)j;
            forward_xor(m);
            std::cout << i << " ^ " << j << " = " << m.a2.at(0, 0) << '\n';
        }
    }
    #endif

    return 0;
}
