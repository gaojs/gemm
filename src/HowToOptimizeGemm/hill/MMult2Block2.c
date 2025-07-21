#include <immintrin.h>

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

void dot4x4(int k, double *a, int lda, double *b, int ldb,double *c, int ldc)
{
  __m256d mc0 = _mm256_loadu_pd(&C(0, 0));
  __m256d mc1 = _mm256_loadu_pd(&C(1, 0));
  __m256d mc2 = _mm256_loadu_pd(&C(2, 0));
  __m256d mc3 = _mm256_loadu_pd(&C(3, 0));
  for (int p=0; p<k; p++ ){
    __m256d ma0 = _mm256_set1_pd(A( 0,p ));
    __m256d ma1 = _mm256_set1_pd(A( 1,p ));
    __m256d ma2 = _mm256_set1_pd(A( 2,p ));
    __m256d ma3 = _mm256_set1_pd(A( 3,p ));
    __m256d mb0 = _mm256_loadu_pd(&B(p, 0));
    mc0 = _mm256_fmadd_pd(ma0, mb0, mc0);
    mc1 = _mm256_fmadd_pd(ma1, mb0, mc1);
    mc2 = _mm256_fmadd_pd(ma2, mb0, mc2);
    mc3 = _mm256_fmadd_pd(ma3, mb0, mc3);
  }  
  _mm256_storeu_pd(&C(0, 0), mc0);
  _mm256_storeu_pd(&C(1, 0), mc1);
  _mm256_storeu_pd(&C(2, 0), mc2);
  _mm256_storeu_pd(&C(3, 0), mc3);
}

void kernel( int m, int n, int k, double *a, int lda, 
  double *b, int ldb,double *c, int ldc )
{
  for (int i=0; i<m; i+=4 ){
    for (int j=0; j<n; j+=4 ){
      dot4x4(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
    }
  }  
}

/* Block sizes */
#define kc 64
#define nc 256
#define mc 256

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
  double *b, int ldb,double *c, int ldc )
{
  for (int i=0; i<m; i+=mc){
    int mb = min( m-i, mc);
    for (int p=0; p<k; p+=kc){
      int kb = min( k-p, kc);
      for (int j=0; j<n; j+=nc){
        int nb = min( n-j, nc);
        kernel(mb, nb, kb, &A(i, p), lda, &B(p, j), ldb, &C(i,j), ldc);
      }
    }
  }
}
 
