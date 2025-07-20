#include <immintrin.h>

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

void dot4x4(double *a, int lda, double *b, int ldb,double *c, int ldc)
{
  __m256d ma0 = _mm256_set1_pd(A( 0,0 ));
  __m256d ma1 = _mm256_set1_pd(A( 1,0 ));
  __m256d ma2 = _mm256_set1_pd(A( 2,0 ));
  __m256d ma3 = _mm256_set1_pd(A( 3,0 ));
  __m256d mb0 = _mm256_loadu_pd(&B(0, 0));
  __m256d mc0 = _mm256_loadu_pd(&C(0, 0));
  __m256d mc1 = _mm256_loadu_pd(&C(1, 0));
  __m256d mc2 = _mm256_loadu_pd(&C(2, 0));
  __m256d mc3 = _mm256_loadu_pd(&C(3, 0));
  _mm256_storeu_pd(&C(0, 0), _mm256_fmadd_pd(ma0, mb0, mc0));
  _mm256_storeu_pd(&C(1, 0), _mm256_fmadd_pd(ma1, mb0, mc1));
  _mm256_storeu_pd(&C(2, 0), _mm256_fmadd_pd(ma2, mb0, mc2));
  _mm256_storeu_pd(&C(3, 0), _mm256_fmadd_pd(ma3, mb0, mc3));
  
}

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
  double *b, int ldb,double *c, int ldc )
{
  for (int p=0; p<k; p++ ){
    for (int i=0; i<m; i+=4 ){
      for (int j=0; j<n; j+=4 ){
        dot4x4(&A(i,p), lda, &B(p,j), ldb, &C(i,j), ldc);
      }
    }
  }
}
 
