#include <cstdlib>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

#include <mulled/matmul.hpp>

struct Fixture : benchmark::Fixture {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    int n = 0;
    Matrix A;
    Matrix B;
    Matrix C;

    void SetUp(const benchmark::State& state) override {
        n = static_cast<int>(state.range(0));
        std::srand(42);
        A = Matrix::Random(n, n);
        B = Matrix::Random(n, n);
        C = Matrix(n, n);
    }

    mulled::matrix_view a_view() const {
        return mulled::matrix_view(A.data(), n, n);
    }

    mulled::matrix_view b_view() const {
        return mulled::matrix_view(B.data(), n, n);
    }

    mulled::mutable_matrix_view c_view() {
        return mulled::mutable_matrix_view(C.data(), n, n);
    }

    void add_counters(benchmark::State& state) {
        state.counters["test1"] = C(0, 0);
        state.counters["test2"] = C(std::min(13, n - 1), std::min(42, n - 1));
    }
};

#define SIZES RangeMultiplier(2)->Range(128, 2048)

BENCHMARK_DEFINE_F(Fixture, matmul_blas)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_blas(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_blas)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_eigen)(benchmark::State& state) {
    for (auto _ : state) {
        C.noalias() = A * B;
        benchmark::DoNotOptimize(C);
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_eigen)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_v1)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v1(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
// v1 is O(n^3) with terrible cache behavior — skip the largest sizes.
BENCHMARK_REGISTER_F(Fixture, matmul_v1)->RangeMultiplier(2)->Range(128, 1024);

BENCHMARK_DEFINE_F(Fixture, matmul_v2)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v2(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_v2)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_v3)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v3(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_v3)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_v4)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v4(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_v4)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_v5)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v5(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_v5)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_v6)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v6(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_v6)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_v7)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v7(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_v7)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_v8)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v8(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_v8)->SIZES;

BENCHMARK_DEFINE_F(Fixture, matmul_v9)(benchmark::State& state) {
    for (auto _ : state) {
        mulled::matmul_v9(a_view(), b_view(), c_view());
        benchmark::DoNotOptimize(C.data());
    }
    add_counters(state);
}
BENCHMARK_REGISTER_F(Fixture, matmul_v9)->SIZES;
