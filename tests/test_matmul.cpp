#include <mulled/matmul.hpp>

#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;

struct Fixture {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    static constexpr int rows = 321;
    static constexpr int cols = 123;
    static constexpr int inner = 42;

    Matrix A = Matrix::Random(rows, inner);
    Matrix B = Matrix::Random(inner, cols);
    Matrix expected = A * B;
    Matrix C = Matrix(rows, cols);

    mulled::matrix_view a_view() const {
        return mulled::matrix_view(A.data(), rows, inner);
    }

    mulled::matrix_view b_view() const {
        return mulled::matrix_view(B.data(), inner, cols);
    }

    mulled::mutable_matrix_view c_view() {
        return mulled::mutable_matrix_view(C.data(), rows, cols);
    }

    void check() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                CAPTURE(i, j);
                REQUIRE_THAT(C(i, j), WithinRel(expected(i, j), 1e-12));
            }
        }
    }
};

TEST_CASE_METHOD(Fixture, "matmul_blas matches Eigen", "[matmul]") {
    mulled::matmul_blas(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v1 matches Eigen", "[matmul]") {
    mulled::matmul_v1(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v2 matches Eigen", "[matmul]") {
    mulled::matmul_v2(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v3 matches Eigen", "[matmul]") {
    mulled::matmul_v3(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v4 matches Eigen", "[matmul]") {
    mulled::matmul_v4(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v5 matches Eigen", "[matmul]") {
    mulled::matmul_v5(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v6 matches Eigen", "[matmul]") {
    mulled::matmul_v6(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v7 matches Eigen", "[matmul]") {
    mulled::matmul_v7(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v8 matches Eigen", "[matmul]") {
    mulled::matmul_v8(a_view(), b_view(), c_view());
    check();
}

TEST_CASE_METHOD(Fixture, "matmul_v9 matches Eigen", "[matmul]") {
    mulled::matmul_v9(a_view(), b_view(), c_view());
    check();
}

