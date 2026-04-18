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

// v5: v4 + cache blocking on k.
// Stream through K in chunks, keeping B rows hot in L2.
// C is loaded/stored across k-blocks (extra traffic for C, but big win for B reuse on large matrices).
static const uniform int KB = 256;

export void matmul_v5_ispc(const uniform double A[],
                           const uniform double B[],
                           uniform double C[],
                           uniform int rows,
                           uniform int cols,
                           uniform int inner) {
    // Zero C up front so we can accumulate across k-blocks.
    for (uniform int i = 0; i < rows; ++i) {
        foreach (j = 0 ... cols) {
            C[i * cols + j] = 0.0;
        }
    }

    for (uniform int kk = 0; kk < inner; kk += KB) {
        uniform int k_end = min(kk + KB, inner);

        uniform int i = 0;
        for (; i + TILE_I <= rows; i += TILE_I) {
            foreach (j = 0 ... cols) {
                double c[TILE_I];

                for (uniform int ti = 0; ti < TILE_I; ++ti) {
                    c[ti] = C[(i + ti) * cols + j];
                }

                for (uniform int k = kk; k < k_end; ++k) {
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
                double c0 = C[i * cols + j];
                for (uniform int k = kk; k < k_end; ++k) {
                    c0 += A[i * inner + k] * B[k * cols + j];
                }
                C[i * cols + j] = c0;
            }
        }
    }
}

// v6: v4 + pack B into j-panels for sequential access.
// For each j-panel, copy B[:, j:j+JP] into contiguous buffer laid out as [k][j_panel].
// The micro-kernel then reads B strictly sequentially.
static const uniform int JP = 128;

export void matmul_v6_ispc(const uniform double A[],
                           const uniform double B[],
                           uniform double C[],
                           uniform int rows,
                           uniform int cols,
                           uniform int inner) {
    uniform double * uniform B_pack = uniform new uniform double[JP * inner];

    for (uniform int jj = 0; jj < cols; jj += JP) {
        uniform int j_end = min(jj + JP, cols);
        uniform int jp_width = j_end - jj;

        // Pack B[:, jj:j_end] into B_pack laid out as [k][j_offset].
        for (uniform int k = 0; k < inner; ++k) {
            foreach (jo = 0 ... jp_width) {
                B_pack[k * jp_width + jo] = B[k * cols + (jj + jo)];
            }
        }

        uniform int i = 0;
        for (; i + TILE_I <= rows; i += TILE_I) {
            foreach (jo = 0 ... jp_width) {
                double c[TILE_I];

                for (uniform int ti = 0; ti < TILE_I; ++ti) {
                    c[ti] = 0.0;
                }

                for (uniform int k = 0; k < inner; ++k) {
                    for (uniform int ti = 0; ti < TILE_I; ++ti) {
                        c[ti] += A[(i + ti) * inner + k] * B_pack[k * jp_width + jo];
                    }
                }

                for (uniform int ti = 0; ti < TILE_I; ++ti) {
                    C[(i + ti) * cols + (jj + jo)] = c[ti];
                }
            }
        }

        for (; i < rows; ++i) {
            foreach (jo = 0 ... jp_width) {
                double c0 = 0.0;
                for (uniform int k = 0; k < inner; ++k) {
                    c0 += A[i * inner + k] * B_pack[k * jp_width + jo];
                }
                C[i * cols + (jj + jo)] = c0;
            }
        }
    }

    delete[] B_pack;
}

// v7: v4 + multiple j-blocks per iteration.
// Process 2 SIMD-width j-blocks at once to increase independent FMA chains in flight.
// TILE_I=4 here keeps total register pressure similar to v4 (4*2 = 8 accumulator vectors).
static const uniform int TILE_I_V7 = 4;

export void matmul_v7_ispc(const uniform double A[],
                           const uniform double B[],
                           uniform double C[],
                           uniform int rows,
                           uniform int cols,
                           uniform int inner) {
    uniform int i = 0;
    for (; i + TILE_I_V7 <= rows; i += TILE_I_V7) {
        uniform int j_chunk = 2 * programCount;
        uniform int jj = 0;

        for (; jj + j_chunk <= cols; jj += j_chunk) {
            double cA[TILE_I_V7];
            double cB[TILE_I_V7];

            for (uniform int ti = 0; ti < TILE_I_V7; ++ti) {
                cA[ti] = 0.0;
                cB[ti] = 0.0;
            }

            for (uniform int k = 0; k < inner; ++k) {
                double bA = B[k * cols + (jj + programIndex)];
                double bB = B[k * cols + (jj + programCount + programIndex)];
                for (uniform int ti = 0; ti < TILE_I_V7; ++ti) {
                    double a_val = A[(i + ti) * inner + k];
                    cA[ti] += a_val * bA;
                    cB[ti] += a_val * bB;
                }
            }

            for (uniform int ti = 0; ti < TILE_I_V7; ++ti) {
                C[(i + ti) * cols + (jj + programIndex)] = cA[ti];
                C[(i + ti) * cols + (jj + programCount + programIndex)] = cB[ti];
            }
        }

        // Remainder j using standard TILE_I_V7 kernel.
        foreach (j = jj ... cols) {
            double c[TILE_I_V7];
            for (uniform int ti = 0; ti < TILE_I_V7; ++ti) {
                c[ti] = 0.0;
            }
            for (uniform int k = 0; k < inner; ++k) {
                for (uniform int ti = 0; ti < TILE_I_V7; ++ti) {
                    c[ti] += A[(i + ti) * inner + k] * B[k * cols + j];
                }
            }
            for (uniform int ti = 0; ti < TILE_I_V7; ++ti) {
                C[(i + ti) * cols + j] = c[ti];
            }
        }
    }

    // Remainder i rows.
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

// v8: v4 + prefetching B rows ahead.
export void matmul_v8_ispc(const uniform double A[],
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
                // Prefetch future B rows into L1.
                if (k + 8 < inner) {
                    prefetch_l1(&B[(k + 8) * cols + j]);
                }
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

// v9: v4 + alignment assumptions + unmasked inner body.
// Requires A/B/C to be 64-byte aligned and cols divisible by programCount.
// Eigen 64-byte aligns; caller is responsible for cols divisibility.
export void matmul_v9_ispc(const uniform double A[],
                           const uniform double B[],
                           uniform double C[],
                           uniform int rows,
                           uniform int cols,
                           uniform int inner) {
    assume(((uniform int64)A & 63) == 0);
    assume(((uniform int64)B & 63) == 0);
    assume(((uniform int64)C & 63) == 0);

    uniform int i = 0;
    for (; i + TILE_I <= rows; i += TILE_I) {
        uniform int j_main = cols - (cols % programCount);
        for (uniform int jj = 0; jj < j_main; jj += programCount) {
            unmasked {
                varying int j = jj + programIndex;
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

        // j remainder with masked kernel.
        foreach (j = j_main ... cols) {
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
