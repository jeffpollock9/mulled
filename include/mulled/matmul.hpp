#pragma once

#include <cblas.h>
#include <experimental/mdspan>
#include <matmul_ispc.h>

namespace mulled {

namespace stdx = std::experimental;

using matrix_view = stdx::mdspan<const double, stdx::dextents<int, 2>>;
using mutable_matrix_view = stdx::mdspan<double, stdx::dextents<int, 2>>;

inline void matmul_blas(matrix_view A, matrix_view B, mutable_matrix_view C) {
    int M = A.extent(0);
    int N = B.extent(1);
    int K = A.extent(1);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                1.0,
                A.data_handle(),
                K,
                B.data_handle(),
                N,
                0.0,
                C.data_handle(),
                N);
}

inline void matmul_v1(matrix_view A, matrix_view B, mutable_matrix_view C) {
    for (int i = 0; i < C.extent(0); ++i) {
        for (int j = 0; j < C.extent(1); ++j) {
            double sum = 0.0;
            for (int k = 0; k < A.extent(1); ++k) {
                sum += A[i, k] * B[k, j];
            }
            C[i, j] = sum;
        }
    }
}

inline void matmul_v2(matrix_view A, matrix_view B, mutable_matrix_view C) {
    for (int i = 0; i < C.extent(0); ++i) {
        for (int j = 0; j < C.extent(1); ++j) {
            C[i, j] = 0.0;
        }
        for (int k = 0; k < A.extent(1); ++k) {
            for (int j = 0; j < B.extent(1); ++j) {
                C[i, j] += A[i, k] * B[k, j];
            }
        }
    }
}

inline void matmul_v3(matrix_view A, matrix_view B, mutable_matrix_view C) {
    ispc::matmul_v3_ispc(A.data_handle(), B.data_handle(), C.data_handle(), A.extent(0), B.extent(1), A.extent(1));
}

inline void matmul_v4(matrix_view A, matrix_view B, mutable_matrix_view C) {
    ispc::matmul_v4_ispc(A.data_handle(), B.data_handle(), C.data_handle(), A.extent(0), B.extent(1), A.extent(1));
}

} // namespace mulled
