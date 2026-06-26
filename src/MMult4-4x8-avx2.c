 #include <stdlib.h>
 #include <stddef.h>

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Block sizes */
#define nc 128
#define kc 256
#define ma 2048

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */
void AddDot4x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
void InnerKernel( int m, int n, int k, double *a, int lda,
    double *b, int ldb, double *c, int ldc, int first_time );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );
void PackMatrixB4( int, double *, int, double * );

void MY_MMult( int m, int n, int k, double *a, int lda,
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int p=0; p<k; p+=kc ){
    int kb = min( k-p, kc );
    for (int j=0; j<n; j+=nc ){
      int nb = min( n-j, nc );
      InnerKernel( m, nb, kb, &A( 0, p), lda, &B( p, j ), ldb, &C( 0, j ), ldc, j==0 );
    }
  }
}

void InnerKernel( int m, int n, int k, double *a, int lda,
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time )
{
  static double *packedA = NULL;
  static size_t packedA_capacity = 0;
  static double *packedB = NULL;
  static size_t packedB_capacity = 0;

  size_t packedA_needed = (size_t)m * (size_t)k * sizeof(double);
  size_t packedB_needed = (size_t)k * (size_t)n * sizeof(double);

  if (first_time) {
    if (packedA == NULL || packedA_needed > packedA_capacity) {
      free(packedA);
      packedA = (double *)aligned_alloc(64, (packedA_needed + 63) & ~((size_t)63));
      packedA_capacity = (packedA_needed + 63) & ~((size_t)63);
    }
  }

  if (packedB == NULL || packedB_needed > packedB_capacity) {
    free(packedB);
    packedB = (double *)aligned_alloc(64, (packedB_needed + 63) & ~((size_t)63));
    packedB_capacity = (packedB_needed + 63) & ~((size_t)63);
  }

  for (int i=0; i<m; i+=4 ){
    if(first_time)
      PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
    for (int j=0; j<n; j+=8 ){
      int nb = min( n-j, 8 );
      if (nb == 8) {
        if(i==0)
          PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
        AddDot4x8( k, &packedA[ i*k ], k , &packedB[ j*k ], 8, &C( i, j ), ldc );
      } else {
        double packedB4[ k * 4 ];
        PackMatrixB4( k, &B( 0, j ), ldb, packedB4 );
        AddDot4x4( k, &packedA[ i*k ], k , packedB4, 4, &C( i, j ), ldc );
      }
    }
  }
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
    double *b_ji_pntr = &B( i, 0 );

    *b_to     = *b_ji_pntr;
    *(b_to+1) = *(b_ji_pntr+1);
    *(b_to+2) = *(b_ji_pntr+2);
    *(b_to+3) = *(b_ji_pntr+3);
    *(b_to+4) = *(b_ji_pntr+4);
    *(b_to+5) = *(b_ji_pntr+5);
    *(b_to+6) = *(b_ji_pntr+6);
    *(b_to+7) = *(b_ji_pntr+7);

    b_to += 8;
  }
}

void PackMatrixB4( int k, double *b, int ldb, double *b_to )
{
  for(int i=0; i<k; i++){
    double *b_ji_pntr = &B( i, 0 );

    *b_to     = *b_ji_pntr;
    *(b_to+1) = *(b_ji_pntr+1);
    *(b_to+2) = *(b_ji_pntr+2);
    *(b_to+3) = *(b_ji_pntr+3);

    b_to += 4;
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

void AddDot4x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  v4df_t
    c_00_c_03_vreg, c_10_c_13_vreg, c_20_c_23_vreg, c_30_c_33_vreg,
    c_04_c_07_vreg, c_14_c_17_vreg, c_24_c_27_vreg, c_34_c_37_vreg,
    b_p0_b_p3_vreg, b_p4_b_p7_vreg,
    a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg;

  c_00_c_03_vreg.v = _mm256_loadu_pd( &C( 0, 0 ) );
  c_10_c_13_vreg.v = _mm256_loadu_pd( &C( 1, 0 ) );
  c_20_c_23_vreg.v = _mm256_loadu_pd( &C( 2, 0 ) );
  c_30_c_33_vreg.v = _mm256_loadu_pd( &C( 3, 0 ) );
  c_04_c_07_vreg.v = _mm256_loadu_pd( &C( 0, 4 ) );
  c_14_c_17_vreg.v = _mm256_loadu_pd( &C( 1, 4 ) );
  c_24_c_27_vreg.v = _mm256_loadu_pd( &C( 2, 4 ) );
  c_34_c_37_vreg.v = _mm256_loadu_pd( &C( 3, 4 ) );

  for (int p=0; p<k; p++ ){
    b_p0_b_p3_vreg.v = _mm256_loadu_pd( (double *) b );
    b_p4_b_p7_vreg.v = _mm256_loadu_pd( (double *) (b+4) );
    b+=8;

    a_0p_vreg.v = _mm256_broadcast_sd( (double *) a );
    a_1p_vreg.v = _mm256_broadcast_sd( (double *) (a+1) );
    a_2p_vreg.v = _mm256_broadcast_sd( (double *) (a+2) );
    a_3p_vreg.v = _mm256_broadcast_sd( (double *) (a+3) );
    a+=4;

    c_00_c_03_vreg.v = _mm256_fmadd_pd( a_0p_vreg.v, b_p0_b_p3_vreg.v, c_00_c_03_vreg.v );
    c_10_c_13_vreg.v = _mm256_fmadd_pd( a_1p_vreg.v, b_p0_b_p3_vreg.v, c_10_c_13_vreg.v );
    c_20_c_23_vreg.v = _mm256_fmadd_pd( a_2p_vreg.v, b_p0_b_p3_vreg.v, c_20_c_23_vreg.v );
    c_30_c_33_vreg.v = _mm256_fmadd_pd( a_3p_vreg.v, b_p0_b_p3_vreg.v, c_30_c_33_vreg.v );

    c_04_c_07_vreg.v = _mm256_fmadd_pd( a_0p_vreg.v, b_p4_b_p7_vreg.v, c_04_c_07_vreg.v );
    c_14_c_17_vreg.v = _mm256_fmadd_pd( a_1p_vreg.v, b_p4_b_p7_vreg.v, c_14_c_17_vreg.v );
    c_24_c_27_vreg.v = _mm256_fmadd_pd( a_2p_vreg.v, b_p4_b_p7_vreg.v, c_24_c_27_vreg.v );
    c_34_c_37_vreg.v = _mm256_fmadd_pd( a_3p_vreg.v, b_p4_b_p7_vreg.v, c_34_c_37_vreg.v );
  }

  _mm256_storeu_pd( &C( 0, 0 ), c_00_c_03_vreg.v );
  _mm256_storeu_pd( &C( 1, 0 ), c_10_c_13_vreg.v );
  _mm256_storeu_pd( &C( 2, 0 ), c_20_c_23_vreg.v );
  _mm256_storeu_pd( &C( 3, 0 ), c_30_c_33_vreg.v );
  _mm256_storeu_pd( &C( 0, 4 ), c_04_c_07_vreg.v );
  _mm256_storeu_pd( &C( 1, 4 ), c_14_c_17_vreg.v );
  _mm256_storeu_pd( &C( 2, 4 ), c_24_c_27_vreg.v );
  _mm256_storeu_pd( &C( 3, 4 ), c_34_c_37_vreg.v );
}

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  v4df_t
    c_00_c_03_vreg, c_10_c_13_vreg, c_20_c_23_vreg, c_30_c_33_vreg,
    b_p0_b_p3_vreg,
    a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg;

  c_00_c_03_vreg.v = _mm256_loadu_pd( &C( 0, 0 ) );
  c_10_c_13_vreg.v = _mm256_loadu_pd( &C( 1, 0 ) );
  c_20_c_23_vreg.v = _mm256_loadu_pd( &C( 2, 0 ) );
  c_30_c_33_vreg.v = _mm256_loadu_pd( &C( 3, 0 ) );

  for (int p=0; p<k; p++ ){
    b_p0_b_p3_vreg.v = _mm256_loadu_pd( (double *) b );
    b+=4;

    a_0p_vreg.v = _mm256_broadcast_sd( (double *) a );
    a_1p_vreg.v = _mm256_broadcast_sd( (double *) (a+1) );
    a_2p_vreg.v = _mm256_broadcast_sd( (double *) (a+2) );
    a_3p_vreg.v = _mm256_broadcast_sd( (double *) (a+3) );
    a+=4;

    c_00_c_03_vreg.v = _mm256_fmadd_pd( a_0p_vreg.v, b_p0_b_p3_vreg.v, c_00_c_03_vreg.v );
    c_10_c_13_vreg.v = _mm256_fmadd_pd( a_1p_vreg.v, b_p0_b_p3_vreg.v, c_10_c_13_vreg.v );
    c_20_c_23_vreg.v = _mm256_fmadd_pd( a_2p_vreg.v, b_p0_b_p3_vreg.v, c_20_c_23_vreg.v );
    c_30_c_33_vreg.v = _mm256_fmadd_pd( a_3p_vreg.v, b_p0_b_p3_vreg.v, c_30_c_33_vreg.v );
  }

  _mm256_storeu_pd( &C( 0, 0 ), c_00_c_03_vreg.v );
  _mm256_storeu_pd( &C( 1, 0 ), c_10_c_13_vreg.v );
  _mm256_storeu_pd( &C( 2, 0 ), c_20_c_23_vreg.v );
  _mm256_storeu_pd( &C( 3, 0 ), c_30_c_33_vreg.v );
}
