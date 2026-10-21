#include <stddef.h>
#include <stdlib.h>

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define nc 72
#define kc 512
#define ma 2048
#define NR 24
#define MR 8

#define min( i, j ) ( (i)<(j) ? (i): (j) )

void AddDot8x24( int k, double *a, int lda, double *b, int ldb, double *c, int ldc );
void InnerKernel( int m, int n, int k, double *a, int lda, 
    double *b, int ldb, double *c, int ldc, int first_time );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, int, double *, int, double * );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int p=0; p<k; p+=kc ){
    int kb = min( k-p, kc );
    for (int j=0; j<n; j+=nc ){
      int nb = min( n-j, nc );
      InnerKernel( m, nb, kb, &A( 0, p), lda, &B(p, j ), ldb, &C( 0,j ), ldc, j==0);
    }
  }
}

void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time )
{
  double *packedA = (double*)aligned_alloc(64, MR * k * sizeof(double));
  double *packedB = (double*)aligned_alloc(64, k * NR * sizeof(double));
  
  int i;
  for (i=0; i + MR <= m; i+=MR ){
    PackMatrixA( k, &A( i, 0 ), lda, packedA );
    int j;
    for (j=0; j + NR <= n; j+=NR ){
      PackMatrixB( k, NR, &B( 0, j ), ldb, packedB );
      AddDot8x24( k, packedA, k, packedB, NR, &C( i,j ), ldc );
    }
    if (j < n) {
      int remaining = n - j;
      PackMatrixB( k, remaining, &B( 0, j ), ldb, packedB );
      for (int p=0; p<k; p++) {
        for (int ii=0; ii<MR; ii++) {
          double a_val = packedA[p*MR + ii];
          for (int jj=0; jj<remaining; jj++) {
            C(i+ii, j+jj) += a_val * packedB[p*NR + jj];
          }
        }
      }
    }
  }
  if (i < m) {
    int remaining_rows = m - i;
    int j;
    for (j=0; j + NR <= n; j+=NR ){
      PackMatrixB( k, NR, &B( 0, j ), ldb, packedB );
      for (int p=0; p<k; p++) {
        for (int ii=0; ii<remaining_rows; ii++) {
          double a_val = A(i+ii, p);
          for (int jj=0; jj<NR; jj++) {
            C(i+ii, j+jj) += a_val * packedB[p*NR + jj];
          }
        }
      }
    }
    if (j < n) {
      int remaining = n - j;
      PackMatrixB( k, remaining, &B( 0, j ), ldb, packedB );
      for (int p=0; p<k; p++) {
        for (int ii=0; ii<remaining_rows; ii++) {
          double a_val = A(i+ii, p);
          for (int jj=0; jj<remaining; jj++) {
            C(i+ii, j+jj) += a_val * packedB[p*NR + jj];
          }
        }
      }
    }
  }
  
  free(packedB);
  free(packedA);
}

void PackMatrixA( int k, double *a, int lda, double *a_to )
{
  double 
    *a_0i_pntr = &A( 0, 0 ), *a_1i_pntr = &A( 1, 0 ),
    *a_2i_pntr = &A( 2, 0 ), *a_3i_pntr = &A( 3, 0 ),
    *a_4i_pntr = &A( 4, 0 ), *a_5i_pntr = &A( 5, 0 ),
    *a_6i_pntr = &A( 6, 0 ), *a_7i_pntr = &A( 7, 0 );

  for(int j=0; j<k; j++){
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
    *a_to++ = *a_4i_pntr++;
    *a_to++ = *a_5i_pntr++;
    *a_to++ = *a_6i_pntr++;
    *a_to++ = *a_7i_pntr++;
  }
}

void PackMatrixB( int k, int ncols, double *b, int ldb, double *b_to )
{
  for(int i=0; i<k; i++){
    double *b_ji_pntr = &B( i , 0 );
    for(int p=0; p<ncols; p++){
        *(b_to + p) = *(b_ji_pntr + p);
    }
    for(int p=ncols; p<NR; p++){
        *(b_to + p) = 0.0;
    }
    b_to += NR;
  }
}

#include <immintrin.h>

void AddDot8x24( int k, double *a, int lda, double *b, int ldb, double *c, int ldc )
{
  __m512d c0_07 = _mm512_setzero_pd();
  __m512d c0_8f = _mm512_setzero_pd();
  __m512d c0_10_17 = _mm512_setzero_pd();
  __m512d c1_07 = _mm512_setzero_pd();
  __m512d c1_8f = _mm512_setzero_pd();
  __m512d c1_10_17 = _mm512_setzero_pd();
  __m512d c2_07 = _mm512_setzero_pd();
  __m512d c2_8f = _mm512_setzero_pd();
  __m512d c2_10_17 = _mm512_setzero_pd();
  __m512d c3_07 = _mm512_setzero_pd();
  __m512d c3_8f = _mm512_setzero_pd();
  __m512d c3_10_17 = _mm512_setzero_pd();
  __m512d c4_07 = _mm512_setzero_pd();
  __m512d c4_8f = _mm512_setzero_pd();
  __m512d c4_10_17 = _mm512_setzero_pd();
  __m512d c5_07 = _mm512_setzero_pd();
  __m512d c5_8f = _mm512_setzero_pd();
  __m512d c5_10_17 = _mm512_setzero_pd();
  __m512d c6_07 = _mm512_setzero_pd();
  __m512d c6_8f = _mm512_setzero_pd();
  __m512d c6_10_17 = _mm512_setzero_pd();
  __m512d c7_07 = _mm512_setzero_pd();
  __m512d c7_8f = _mm512_setzero_pd();
  __m512d c7_10_17 = _mm512_setzero_pd();

  for (int p=0; p<k; p++){
    __m512d a0 = _mm512_broadcastsd_pd(_mm_load_sd(a));
    __m512d a1 = _mm512_broadcastsd_pd(_mm_load_sd(a+1));
    __m512d a2 = _mm512_broadcastsd_pd(_mm_load_sd(a+2));
    __m512d a3 = _mm512_broadcastsd_pd(_mm_load_sd(a+3));
    __m512d a4 = _mm512_broadcastsd_pd(_mm_load_sd(a+4));
    __m512d a5 = _mm512_broadcastsd_pd(_mm_load_sd(a+5));
    __m512d a6 = _mm512_broadcastsd_pd(_mm_load_sd(a+6));
    __m512d a7 = _mm512_broadcastsd_pd(_mm_load_sd(a+7));
    a += MR;

    __m512d b0_7 = _mm512_loadu_pd(b);
    __m512d b8_f = _mm512_loadu_pd(b + 8);
    __m512d b10_17 = _mm512_loadu_pd(b + 16);
    b += NR;

    c0_07 = _mm512_fmadd_pd(a0, b0_7, c0_07);
    c0_8f = _mm512_fmadd_pd(a0, b8_f, c0_8f);
    c0_10_17 = _mm512_fmadd_pd(a0, b10_17, c0_10_17);
    c1_07 = _mm512_fmadd_pd(a1, b0_7, c1_07);
    c1_8f = _mm512_fmadd_pd(a1, b8_f, c1_8f);
    c1_10_17 = _mm512_fmadd_pd(a1, b10_17, c1_10_17);
    c2_07 = _mm512_fmadd_pd(a2, b0_7, c2_07);
    c2_8f = _mm512_fmadd_pd(a2, b8_f, c2_8f);
    c2_10_17 = _mm512_fmadd_pd(a2, b10_17, c2_10_17);
    c3_07 = _mm512_fmadd_pd(a3, b0_7, c3_07);
    c3_8f = _mm512_fmadd_pd(a3, b8_f, c3_8f);
    c3_10_17 = _mm512_fmadd_pd(a3, b10_17, c3_10_17);
    c4_07 = _mm512_fmadd_pd(a4, b0_7, c4_07);
    c4_8f = _mm512_fmadd_pd(a4, b8_f, c4_8f);
    c4_10_17 = _mm512_fmadd_pd(a4, b10_17, c4_10_17);
    c5_07 = _mm512_fmadd_pd(a5, b0_7, c5_07);
    c5_8f = _mm512_fmadd_pd(a5, b8_f, c5_8f);
    c5_10_17 = _mm512_fmadd_pd(a5, b10_17, c5_10_17);
    c6_07 = _mm512_fmadd_pd(a6, b0_7, c6_07);
    c6_8f = _mm512_fmadd_pd(a6, b8_f, c6_8f);
    c6_10_17 = _mm512_fmadd_pd(a6, b10_17, c6_10_17);
    c7_07 = _mm512_fmadd_pd(a7, b0_7, c7_07);
    c7_8f = _mm512_fmadd_pd(a7, b8_f, c7_8f);
    c7_10_17 = _mm512_fmadd_pd(a7, b10_17, c7_10_17);
  }

  _mm512_storeu_pd(&C(0,0), _mm512_add_pd(_mm512_loadu_pd(&C(0,0)), c0_07));
  _mm512_storeu_pd(&C(0,8), _mm512_add_pd(_mm512_loadu_pd(&C(0,8)), c0_8f));
  _mm512_storeu_pd(&C(0,16), _mm512_add_pd(_mm512_loadu_pd(&C(0,16)), c0_10_17));
  _mm512_storeu_pd(&C(1,0), _mm512_add_pd(_mm512_loadu_pd(&C(1,0)), c1_07));
  _mm512_storeu_pd(&C(1,8), _mm512_add_pd(_mm512_loadu_pd(&C(1,8)), c1_8f));
  _mm512_storeu_pd(&C(1,16), _mm512_add_pd(_mm512_loadu_pd(&C(1,16)), c1_10_17));
  _mm512_storeu_pd(&C(2,0), _mm512_add_pd(_mm512_loadu_pd(&C(2,0)), c2_07));
  _mm512_storeu_pd(&C(2,8), _mm512_add_pd(_mm512_loadu_pd(&C(2,8)), c2_8f));
  _mm512_storeu_pd(&C(2,16), _mm512_add_pd(_mm512_loadu_pd(&C(2,16)), c2_10_17));
  _mm512_storeu_pd(&C(3,0), _mm512_add_pd(_mm512_loadu_pd(&C(3,0)), c3_07));
  _mm512_storeu_pd(&C(3,8), _mm512_add_pd(_mm512_loadu_pd(&C(3,8)), c3_8f));
  _mm512_storeu_pd(&C(3,16), _mm512_add_pd(_mm512_loadu_pd(&C(3,16)), c3_10_17));
  _mm512_storeu_pd(&C(4,0), _mm512_add_pd(_mm512_loadu_pd(&C(4,0)), c4_07));
  _mm512_storeu_pd(&C(4,8), _mm512_add_pd(_mm512_loadu_pd(&C(4,8)), c4_8f));
  _mm512_storeu_pd(&C(4,16), _mm512_add_pd(_mm512_loadu_pd(&C(4,16)), c4_10_17));
  _mm512_storeu_pd(&C(5,0), _mm512_add_pd(_mm512_loadu_pd(&C(5,0)), c5_07));
  _mm512_storeu_pd(&C(5,8), _mm512_add_pd(_mm512_loadu_pd(&C(5,8)), c5_8f));
  _mm512_storeu_pd(&C(5,16), _mm512_add_pd(_mm512_loadu_pd(&C(5,16)), c5_10_17));
  _mm512_storeu_pd(&C(6,0), _mm512_add_pd(_mm512_loadu_pd(&C(6,0)), c6_07));
  _mm512_storeu_pd(&C(6,8), _mm512_add_pd(_mm512_loadu_pd(&C(6,8)), c6_8f));
  _mm512_storeu_pd(&C(6,16), _mm512_add_pd(_mm512_loadu_pd(&C(6,16)), c6_10_17));
  _mm512_storeu_pd(&C(7,0), _mm512_add_pd(_mm512_loadu_pd(&C(7,0)), c7_07));
  _mm512_storeu_pd(&C(7,8), _mm512_add_pd(_mm512_loadu_pd(&C(7,8)), c7_8f));
  _mm512_storeu_pd(&C(7,16), _mm512_add_pd(_mm512_loadu_pd(&C(7,16)), c7_10_17));
}