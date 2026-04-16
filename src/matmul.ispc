export void matmul_v3_ispc(const uniform double A[],
                           const uniform double B[],
                           uniform double C[],
                           uniform int rows,
                           uniform int cols,
                           uniform int inner) {
    for (uniform int i = 0; i < rows; ++i) {
        foreach (j = 0 ... cols) {
            C[i * cols + j] = 0.0;
        }
        for (uniform int k = 0; k < inner; ++k) {
            uniform double Aik = A[i * inner + k];
            foreach (j = 0 ... cols) {
                C[i * cols + j] += Aik * B[k * cols + j];
            }
        }
    }
}

static const uniform int TILE_I = 8;

export void matmul_v4_ispc(const uniform double A[],
                           const uniform double B[],
                           uniform double C[],
                           uniform int rows,
                           uniform int cols,
                           uniform int inner) {
    uniform int i = 0;
    for (; i + TILE_I <= rows; i += TILE_I) {
        foreach (j = 0 ... cols) {
            double c[TILE_I];

            for (uniform int ti = 0; ti < TILE_I; ++ti) {
                c[ti] = 0.0;
            }

            for (uniform int k = 0; k < inner; ++k) {
                for (uniform int ti = 0; ti < TILE_I; ++ti) {
                    c[ti] += A[(i + ti) * inner + k] * B[k * cols + j];
                }
            }

            for (uniform int ti = 0; ti < TILE_I; ++ti) {
                C[(i + ti) * cols + j] = c[ti];
            }
        }
    }

    for (; i < rows; ++i) {
        foreach (j = 0 ... cols) {
            double c0 = 0.0;
            for (uniform int k = 0; k < inner; ++k) {
                c0 += A[i * inner + k] * B[k * cols + j];
            }
            C[i * cols + j] = c0;
        }
    }
}
