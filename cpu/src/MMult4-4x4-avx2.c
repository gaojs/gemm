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
void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
void InnerKernel( int m, int n, int k, double *a, int lda, 
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
      InnerKernel( m, nb, kb, &A( 0, p), lda, &B(p, j ), ldb, &C( 0,j ), ldc, j==0);
    }
  }
}

void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time )
{
  static double packedA[ ma * kc ];
  double packedB[ k * n ];
  for (int i=0; i<m; i+=4 ){ /* Loop over the rows of C */  
    if(first_time)
      PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
    for (int j=0; j<n; j+=4 ){ /* Loop over the columns of C*/
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if(i==0)
      PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
      AddDot4x4( k, &packedA[ i*k ], k , &packedB[j*k], 4, &C( i,j ), ldc );
    }
  }
}

void PackMatrixA( int k, double *a, int lda, double *a_to )
{
  double 
    *a_0i_pntr = &A( 0, 0 ), *a_1i_pntr = &A( 1, 0 ),
    *a_2i_pntr = &A( 2, 0 ), *a_3i_pntr = &A( 3, 0 );

  for(int j=0; j<k; j++){ /* loop over columns of A */
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
  }
}

void PackMatrixB( int k, double *b, int ldb, double *b_to )
{
  for(int i=0; i<k; i++){  /* loop over rows of B */
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

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  v4df_t
    c_00_c_03_vreg, c_10_c_13_vreg, c_20_c_23_vreg, c_30_c_33_vreg,
    b_p0_b_p3_vreg,
    a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg; 

  c_00_c_03_vreg.v = _mm256_setzero_pd();   
  c_10_c_13_vreg.v = _mm256_setzero_pd();
  c_20_c_23_vreg.v = _mm256_setzero_pd(); 
  c_30_c_33_vreg.v = _mm256_setzero_pd(); 

  for (int p=0; p<k; p++ ){
    b_p0_b_p3_vreg.v = _mm256_loadu_pd( (double *) b );
    b+=4;

    a_0p_vreg.v = _mm256_broadcast_sd( (double *) a );   /* load and duplicate */
    a_1p_vreg.v = _mm256_broadcast_sd( (double *) (a+1));   /* load and duplicate */
    a_2p_vreg.v = _mm256_broadcast_sd( (double *) (a+2) );   /* load and duplicate */
    a_3p_vreg.v = _mm256_broadcast_sd( (double *) (a+3) );   /* load and duplicate */
    a+=4;

    c_00_c_03_vreg.v += a_0p_vreg.v * b_p0_b_p3_vreg.v;
    c_10_c_13_vreg.v += a_1p_vreg.v * b_p0_b_p3_vreg.v;
    c_20_c_23_vreg.v += a_2p_vreg.v * b_p0_b_p3_vreg.v;
    c_30_c_33_vreg.v += a_3p_vreg.v * b_p0_b_p3_vreg.v;

  }

  C( 0, 0 ) += c_00_c_03_vreg.d[0];  C( 1, 0 ) += c_10_c_13_vreg.d[0];  
  C( 2, 0 ) += c_20_c_23_vreg.d[0];  C( 3, 0 ) += c_30_c_33_vreg.d[0]; 

  C( 0, 1 ) += c_00_c_03_vreg.d[1];  C( 1, 1 ) += c_10_c_13_vreg.d[1];  
  C( 2, 1 ) += c_20_c_23_vreg.d[1];  C( 3, 1 ) += c_30_c_33_vreg.d[1]; 

  C( 0, 2 ) += c_00_c_03_vreg.d[2];  C( 1, 2 ) += c_10_c_13_vreg.d[2];  
  C( 2, 2 ) += c_20_c_23_vreg.d[2];  C( 3, 2 ) += c_30_c_33_vreg.d[2]; 

  C( 0, 3 ) += c_00_c_03_vreg.d[3];  C( 1, 3 ) += c_10_c_13_vreg.d[3];  
  C( 2, 3 ) += c_20_c_23_vreg.d[3];  C( 3, 3 ) += c_30_c_33_vreg.d[3]; 
}
