/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Block sizes */
#define nc 128
#define kc 256
#define ma 2048
#define MR 8
#define NR 8

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */
void AddDot8x8( int, double *, int, double *, int, double *, int );
void InnerKernel( int m, int n, int k, double *a, int lda, 
    double *b, int ldb, double *c, int ldc, int first_time );
void PackMatrixA( int, int, double *, int, double * );
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
  static double packedA[ ma * kc ];
  double packedB[ kc * nc ];
  int i;
  for (i=0; i + MR <= m; i+=MR ){
    if(first_time)
      PackMatrixA( k, MR, &A( i, 0 ), lda, &packedA[ i*k ] );
    int j;
    for (j=0; j + NR <= n; j+=NR ){
      if(i==0)
      PackMatrixB( k, NR, &B( 0, j ), ldb, &packedB[ j*k ] );
      AddDot8x8( k, &packedA[ i*k ], k , &packedB[j*k], NR, &C( i,j ), ldc );
    }
    if (j < n) {
      int remaining = n - j;
      if(i==0)
      PackMatrixB( k, remaining, &B( 0, j ), ldb, &packedB[ j*k ] );
      for (int p=0; p<k; p++) {
        for (int ii=0; ii<MR; ii++) {
          double a_val = packedA[i*k + p*MR + ii];
          for (int jj=0; jj<remaining; jj++) {
            C(i+ii, j+jj) += a_val * packedB[j*k + p*NR + jj];
          }
        }
      }
    }
  }
  if (i < m) {
    int remaining_m = m - i;
    if(first_time)
      PackMatrixA( k, remaining_m, &A( i, 0 ), lda, &packedA[ i*k ] );
    int j;
    for (j=0; j + NR <= n; j+=NR ){
      if(i==0)
      PackMatrixB( k, NR, &B( 0, j ), ldb, &packedB[ j*k ] );
      for (int p=0; p<k; p++) {
        for (int ii=0; ii<remaining_m; ii++) {
          double a_val = packedA[i*k + p*MR + ii];
          for (int jj=0; jj<NR; jj++) {
            C(i+ii, j+jj) += a_val * packedB[j*k + p*NR + jj];
          }
        }
      }
    }
    if (j < n) {
      int remaining_n = n - j;
      if(i==0)
      PackMatrixB( k, remaining_n, &B( 0, j ), ldb, &packedB[ j*k ] );
      for (int p=0; p<k; p++) {
        for (int ii=0; ii<remaining_m; ii++) {
          double a_val = packedA[i*k + p*MR + ii];
          for (int jj=0; jj<remaining_n; jj++) {
            C(i+ii, j+jj) += a_val * packedB[j*k + p*NR + jj];
          }
        }
      }
    }
  }
}

void PackMatrixA( int k, int nrows, double *a, int lda, double *a_to )
{
  double *a_pntrs[8];
  a_pntrs[0] = &A( 0, 0 );
  if (nrows > 1) a_pntrs[1] = &A( 1, 0 );
  if (nrows > 2) a_pntrs[2] = &A( 2, 0 );
  if (nrows > 3) a_pntrs[3] = &A( 3, 0 );
  if (nrows > 4) a_pntrs[4] = &A( 4, 0 );
  if (nrows > 5) a_pntrs[5] = &A( 5, 0 );
  if (nrows > 6) a_pntrs[6] = &A( 6, 0 );
  if (nrows > 7) a_pntrs[7] = &A( 7, 0 );

  for(int j=0; j<k; j++){
    for (int i=0; i<nrows; i++) {
      *a_to++ = *a_pntrs[i]++;
    }
    for (int i=nrows; i<MR; i++) {
      *a_to++ = 0.0;
    }
  }
}

void PackMatrixB( int k, int ncols, double *b, int ldb, double *b_to )
{
  for(int i=0; i<k; i++){
    double *b_ji_pntr = &B( i, 0 );
    for (int p=0; p<ncols; p++) {
      *(b_to + p) = *(b_ji_pntr + p);
    }
    for (int p=ncols; p<NR; p++) {
      *(b_to + p) = 0.0;
    }
    b_to += NR;
  }
}

#include <immintrin.h>

typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

typedef union 
{
  __m256d v;
  double d[4];
} v4df_t; 


void AddDot8x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{

  __m512d c0 = _mm512_setzero_pd();   
  __m512d c1 = _mm512_setzero_pd();
  __m512d c2 = _mm512_setzero_pd(); 
  __m512d c3 = _mm512_setzero_pd();
  __m512d c4 = _mm512_setzero_pd();   
  __m512d c5 = _mm512_setzero_pd();
  __m512d c6 = _mm512_setzero_pd(); 
  __m512d c7 = _mm512_setzero_pd();   

  for (int p=0; p<k; p++ ){
    __m512d b_vec = _mm512_loadu_pd( b );
    b+=NR;

    __m512d a0 = _mm512_broadcastsd_pd(_mm_load_sd(a) );
    __m512d a1 = _mm512_broadcastsd_pd(_mm_load_sd(a+1) );
    __m512d a2 = _mm512_broadcastsd_pd(_mm_load_sd(a+2) );
    __m512d a3 = _mm512_broadcastsd_pd(_mm_load_sd(a+3) );
    __m512d a4 = _mm512_broadcastsd_pd(_mm_load_sd(a+4) );
    __m512d a5 = _mm512_broadcastsd_pd(_mm_load_sd(a+5) );
    __m512d a6 = _mm512_broadcastsd_pd(_mm_load_sd(a+6) );
    __m512d a7 = _mm512_broadcastsd_pd(_mm_load_sd(a+7) );
    a+=MR;

    c0 = _mm512_fmadd_pd(b_vec, a0, c0);
    c1 = _mm512_fmadd_pd(b_vec, a1, c1);
    c2 = _mm512_fmadd_pd(b_vec, a2, c2);
    c3 = _mm512_fmadd_pd(b_vec, a3, c3);
    c4 = _mm512_fmadd_pd(b_vec, a4, c4);
    c5 = _mm512_fmadd_pd(b_vec, a5, c5);
    c6 = _mm512_fmadd_pd(b_vec, a6, c6);
    c7 = _mm512_fmadd_pd(b_vec, a7, c7);

  }
  _mm512_storeu_pd(&C(0,0), _mm512_add_pd(_mm512_loadu_pd(&C(0,0)), c0));
  _mm512_storeu_pd(&C(1,0), _mm512_add_pd(_mm512_loadu_pd(&C(1,0)), c1));
  _mm512_storeu_pd(&C(2,0), _mm512_add_pd(_mm512_loadu_pd(&C(2,0)), c2));
  _mm512_storeu_pd(&C(3,0), _mm512_add_pd(_mm512_loadu_pd(&C(3,0)), c3));
  _mm512_storeu_pd(&C(4,0), _mm512_add_pd(_mm512_loadu_pd(&C(4,0)), c4));
  _mm512_storeu_pd(&C(5,0), _mm512_add_pd(_mm512_loadu_pd(&C(5,0)), c5));
  _mm512_storeu_pd(&C(6,0), _mm512_add_pd(_mm512_loadu_pd(&C(6,0)), c6));
  _mm512_storeu_pd(&C(7,0), _mm512_add_pd(_mm512_loadu_pd(&C(7,0)), c7));
}
