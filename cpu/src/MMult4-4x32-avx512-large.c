#include <stddef.h>
#include <stdlib.h>

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Block sizes - Optimized for cache */
#define nc 128
#define kc 512
#define ma 2048

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */
void AddDot4x32_large( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
void InnerKernel_large( int m, int n, int k, double *a, int lda,
    double *b, int ldb, double *c, int ldc, int first_time );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );

void MY_MMult( int m, int n, int k, double *a, int lda,
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  /* This time, we compute a m x nc block of C by a call to the InnerKernel */
  for (int p=0; p<k; p+=kc ){
    int kb = min( k-p, kc );
    for (int j=0; j<n; j+=nc ){
      int nb = min( n-j, nc );
      InnerKernel_large( m, nb, kb, &A( 0, p), lda, &B(p, j ), ldb, &C( 0,j ), ldc, j==0);
    }
  }
}

void InnerKernel_large( int m, int n, int k, double *a, int lda,
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time )
{
  static double *packedA = NULL;
  static int max_packedA_size = 0;

  if (first_time) {
    int required_size = m * k * sizeof(double);
    if (packedA == NULL || required_size > max_packedA_size) {
      if (packedA != NULL) free(packedA);
      packedA = (double*)aligned_alloc(64, required_size);
      max_packedA_size = required_size;
    }
  }

  double *packedB = (double*)aligned_alloc(64, k * n * sizeof(double));
  for (int i=0; i<m; i+=4 ){
    if(first_time)
      PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
    for (int j=0; j<n; j+=32 ){
      if( i == 0 )
        PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
      AddDot4x32_large( k, &packedA[ i*k ], k , &packedB[ j*k ], 32, &C( i,j ), ldc );
    }
  }
  free(packedB);
}

void PackMatrixA( int k, double *a, int lda, double *a_to )
{
  double
    *a_0i_pntr = &A( 0, 0 ), *a_1i_pntr = &A( 1, 0 ),
    *a_2i_pntr = &A( 2, 0 ), *a_3i_pntr = &A( 3, 0 );

  for(int j=0; j<k; j++){
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
  }
}

void PackMatrixB( int k, double *b, int ldb, double *b_to )
{
  for(int i=0; i<k; i++){
    double *b_ji_pntr = &B( i , 0 );

    for(int p=0; p<32; p++){
        *(b_to + p) = *(b_ji_pntr + p);
    }
    b_to += 32;
  }
}

#include <immintrin.h>

void AddDot4x32_large( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  __m512d
    c0_0, c0_1, c0_2, c0_3,
    c1_0, c1_1, c1_2, c1_3,
    c2_0, c2_1, c2_2, c2_3,
    c3_0, c3_1, c3_2, c3_3,
    b0, b1, b2, b3,
    a0, a1, a2, a3;

  c0_0 = _mm512_loadu_pd( &C( 0, 0 ) );
  c0_1 = _mm512_loadu_pd( &C( 0, 8 ) );
  c0_2 = _mm512_loadu_pd( &C( 0, 16 ) );
  c0_3 = _mm512_loadu_pd( &C( 0, 24 ) );

  c1_0 = _mm512_loadu_pd( &C( 1, 0 ) );
  c1_1 = _mm512_loadu_pd( &C( 1, 8 ) );
  c1_2 = _mm512_loadu_pd( &C( 1, 16 ) );
  c1_3 = _mm512_loadu_pd( &C( 1, 24 ) );

  c2_0 = _mm512_loadu_pd( &C( 2, 0 ) );
  c2_1 = _mm512_loadu_pd( &C( 2, 8 ) );
  c2_2 = _mm512_loadu_pd( &C( 2, 16 ) );
  c2_3 = _mm512_loadu_pd( &C( 2, 24 ) );

  c3_0 = _mm512_loadu_pd( &C( 3, 0 ) );
  c3_1 = _mm512_loadu_pd( &C( 3, 8 ) );
  c3_2 = _mm512_loadu_pd( &C( 3, 16 ) );
  c3_3 = _mm512_loadu_pd( &C( 3, 24 ) );

  for (int p=0; p<k; p++ ){
    a0 = _mm512_broadcastsd_pd(_mm_load_sd(a));
    a1 = _mm512_broadcastsd_pd(_mm_load_sd(a+1));
    a2 = _mm512_broadcastsd_pd(_mm_load_sd(a+2));
    a3 = _mm512_broadcastsd_pd(_mm_load_sd(a+3));
    a+=4;

    b0 = _mm512_load_pd( (double *) b );
    b1 = _mm512_load_pd( (double *) (b+8) );
    b2 = _mm512_load_pd( (double *) (b+16) );
    b3 = _mm512_load_pd( (double *) (b+24) );
    b+=32;

    c0_0 = _mm512_fmadd_pd(a0, b0, c0_0);
    c1_0 = _mm512_fmadd_pd(a1, b0, c1_0);
    c2_0 = _mm512_fmadd_pd(a2, b0, c2_0);
    c3_0 = _mm512_fmadd_pd(a3, b0, c3_0);

    c0_1 = _mm512_fmadd_pd(a0, b1, c0_1);
    c1_1 = _mm512_fmadd_pd(a1, b1, c1_1);
    c2_1 = _mm512_fmadd_pd(a2, b1, c2_1);
    c3_1 = _mm512_fmadd_pd(a3, b1, c3_1);

    c0_2 = _mm512_fmadd_pd(a0, b2, c0_2);
    c1_2 = _mm512_fmadd_pd(a1, b2, c1_2);
    c2_2 = _mm512_fmadd_pd(a2, b2, c2_2);
    c3_2 = _mm512_fmadd_pd(a3, b2, c3_2);

    c0_3 = _mm512_fmadd_pd(a0, b3, c0_3);
    c1_3 = _mm512_fmadd_pd(a1, b3, c1_3);
    c2_3 = _mm512_fmadd_pd(a2, b3, c2_3);
    c3_3 = _mm512_fmadd_pd(a3, b3, c3_3);
  }

  _mm512_storeu_pd( &C( 0, 0 ), c0_0 );
  _mm512_storeu_pd( &C( 0, 8 ), c0_1 );
  _mm512_storeu_pd( &C( 0, 16 ), c0_2 );
  _mm512_storeu_pd( &C( 0, 24 ), c0_3 );

  _mm512_storeu_pd( &C( 1, 0 ), c1_0 );
  _mm512_storeu_pd( &C( 1, 8 ), c1_1 );
  _mm512_storeu_pd( &C( 1, 16 ), c1_2 );
  _mm512_storeu_pd( &C( 1, 24 ), c1_3 );

  _mm512_storeu_pd( &C( 2, 0 ), c2_0 );
  _mm512_storeu_pd( &C( 2, 8 ), c2_1 );
  _mm512_storeu_pd( &C( 2, 16 ), c2_2 );
  _mm512_storeu_pd( &C( 2, 24 ), c2_3 );

  _mm512_storeu_pd( &C( 3, 0 ), c3_0 );
  _mm512_storeu_pd( &C( 3, 8 ), c3_1 );
  _mm512_storeu_pd( &C( 3, 16 ), c3_2 );
  _mm512_storeu_pd( &C( 3, 24 ), c3_3 );
}