#include <Eigen/Dense>
#include <benchmark/benchmark.h>

#include <mulled/matmul.hpp>

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr int N = 1234;

struct Matrices {
    Matrix A = Matrix::Random(N, N);
    Matrix B = Matrix::Random(N, N);
    Matrix C = Matrix(N, N);

    mulled::matrix_view a_view() const {
        return mulled::matrix_view(A.data(), N, N);
    }

    mulled::matrix_view b_view() const {
        return mulled::matrix_view(B.data(), N, N);
    }

    mulled::mutable_matrix_view c_view() {
        return mulled::mutable_matrix_view(C.data(), N, N);
    }
};

Matrices& matrices() {
    static Matrices instance;
    return instance;
}

void add_counters(benchmark::State& state, const Matrix& C) {
    state.counters["test1"] = C(0, 0);
    state.counters["test2"] = C(13, 42);
}

void BM_matmul_blas(benchmark::State& state) {
    auto& m = matrices();
    for (auto _ : state) {
        mulled::matmul_blas(m.a_view(), m.b_view(), m.c_view());
        benchmark::DoNotOptimize(m.C.data());
    }
    add_counters(state, m.C);
}
BENCHMARK(BM_matmul_blas);

void BM_matmul_eigen(benchmark::State& state) {
    auto& m = matrices();
    for (auto _ : state) {
        m.C.noalias() = m.A * m.B;
        benchmark::DoNotOptimize(m.C);
    }
    add_counters(state, m.C);
}
BENCHMARK(BM_matmul_eigen);

void BM_matmul_v1(benchmark::State& state) {
    auto& m = matrices();
    for (auto _ : state) {
        mulled::matmul_v1(m.a_view(), m.b_view(), m.c_view());
        benchmark::DoNotOptimize(m.C.data());
    }
    add_counters(state, m.C);
}
BENCHMARK(BM_matmul_v1);

void BM_matmul_v2(benchmark::State& state) {
    auto& m = matrices();
    for (auto _ : state) {
        mulled::matmul_v2(m.a_view(), m.b_view(), m.c_view());
        benchmark::DoNotOptimize(m.C.data());
    }
    add_counters(state, m.C);
}
BENCHMARK(BM_matmul_v2);

void BM_matmul_v3(benchmark::State& state) {
    auto& m = matrices();
    for (auto _ : state) {
        mulled::matmul_v3(m.a_view(), m.b_view(), m.c_view());
        benchmark::DoNotOptimize(m.C.data());
    }
    add_counters(state, m.C);
}
BENCHMARK(BM_matmul_v3);

void BM_matmul_v4(benchmark::State& state) {
    auto& m = matrices();
    for (auto _ : state) {
        mulled::matmul_v4(m.a_view(), m.b_view(), m.c_view());
        benchmark::DoNotOptimize(m.C.data());
    }
    add_counters(state, m.C);
}
BENCHMARK(BM_matmul_v4);
