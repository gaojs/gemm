#include <stdlib.h>
#include <stddef.h>
#include <immintrin.h>

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define MC 16
#define NC 16
#define KC 16

#define min( i, j ) ( (i)<(j) ? (i): (j) )

typedef union
{
  __m256d v;
  double d[4];
} v4df_t;

void MY_MMult( int m, int n, int k, double *a, int lda,
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int i = 0; i < m; i += MC) {
    int mb = min(m - i, MC);
    for (int j = 0; j < n; j += NC) {
      int nb = min(n - j, NC);

      for (int ii = i; ii < i + mb; ii += 4) {
        for (int jj = j; jj < j + nb; jj += 8) {
          v4df_t
            c03, c13, c23, c33,
            c47, c57, c67, c77,
            b03, b47,
            a0, a1, a2, a3;

          c03.v = _mm256_loadu_pd(&C(ii+0, jj+0));
          c13.v = _mm256_loadu_pd(&C(ii+1, jj+0));
          c23.v = _mm256_loadu_pd(&C(ii+2, jj+0));
          c33.v = _mm256_loadu_pd(&C(ii+3, jj+0));
          c47.v = _mm256_loadu_pd(&C(ii+0, jj+4));
          c57.v = _mm256_loadu_pd(&C(ii+1, jj+4));
          c67.v = _mm256_loadu_pd(&C(ii+2, jj+4));
          c77.v = _mm256_loadu_pd(&C(ii+3, jj+4));

          for (int p = 0; p < k; p += KC) {
            int kb = min(k - p, KC);

            for (int p2 = p; p2 < p + kb; p2++) {
              b03.v = _mm256_loadu_pd(&B(p2, jj+0));
              b47.v = _mm256_loadu_pd(&B(p2, jj+4));

              a0.v = _mm256_broadcast_sd(&A(ii+0, p2));
              a1.v = _mm256_broadcast_sd(&A(ii+1, p2));
              a2.v = _mm256_broadcast_sd(&A(ii+2, p2));
              a3.v = _mm256_broadcast_sd(&A(ii+3, p2));

              c03.v = _mm256_fmadd_pd(a0.v, b03.v, c03.v);
              c13.v = _mm256_fmadd_pd(a1.v, b03.v, c13.v);
              c23.v = _mm256_fmadd_pd(a2.v, b03.v, c23.v);
              c33.v = _mm256_fmadd_pd(a3.v, b03.v, c33.v);

              c47.v = _mm256_fmadd_pd(a0.v, b47.v, c47.v);
              c57.v = _mm256_fmadd_pd(a1.v, b47.v, c57.v);
              c67.v = _mm256_fmadd_pd(a2.v, b47.v, c67.v);
              c77.v = _mm256_fmadd_pd(a3.v, b47.v, c77.v);
            }
          }

          _mm256_storeu_pd(&C(ii+0, jj+0), c03.v);
          _mm256_storeu_pd(&C(ii+1, jj+0), c13.v);
          _mm256_storeu_pd(&C(ii+2, jj+0), c23.v);
          _mm256_storeu_pd(&C(ii+3, jj+0), c33.v);
          _mm256_storeu_pd(&C(ii+0, jj+4), c47.v);
          _mm256_storeu_pd(&C(ii+1, jj+4), c57.v);
          _mm256_storeu_pd(&C(ii+2, jj+4), c67.v);
          _mm256_storeu_pd(&C(ii+3, jj+4), c77.v);
        }
      }
    }
  }
}