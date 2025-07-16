
/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Block sizes */
#define nc 128
#define kc 256
#define ma 2000

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

void AddDot4x4( int, double *, int, double *, int, double *, int );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );
void InnerKernel( int, int, int, double *, int, double *, int, double *, int, int );


void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int p=0; p<k; p+=kc ){
    int pb = min( k-p, kc );
    for (int i=0; i<n; i+=nc ){
      int ib = min( n-i, nc );
      InnerKernel( m, ib, pb, &A( 0, p), lda, &B(p, i ), ldb, &C( 0,i ), ldc , i == 0 );
    }
  }
}


void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time )
{
  int i, j;
  static double 
    packedA[ ma * kc ];
  double packedB[ k * n ];
  for ( i=0; i<m; i+=4 ){   
    if(first_time)     /* Loop over the columns of C, unrolled by 4 */
      PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
    for ( j=0; j<n; j+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
      AddDot4x4( k, &packedA[ i*k ], k , &packedB[ j*k ], 4 , &C( i,j ), ldc );
    }
  }
}

void PackMatrixA( int k, double *a, int lda, double *a_to )
{
  double 
    *a_0i_pntr = &A( 0, 0 ), *a_1i_pntr = &A( 1, 0 ),
    *a_2i_pntr = &A( 2, 0 ), *a_3i_pntr = &A( 3, 0 );
  int j;

  for( j=0; j<k; j++){  /* loop over columns of A */
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
  }
}

void PackMatrixB( int k, double *b, int ldb, double *b_to )
{
  int i;

  for( i=0; i<k; i++){  /* loop over rows of B */
    double 
      *b_ji_pntr = &B( i , 0 );

    *b_to     = *b_ji_pntr;
    *(b_to+1) = *(b_ji_pntr+1);
    *(b_to+2) = *(b_ji_pntr+2);
    *(b_to+3) = *(b_ji_pntr+3);

    b_to += 4;
  }
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     And now we use vector registers and instructions */


  v2df_t
    c_00_c_01_vreg,    c_10_c_11_vreg,    c_20_c_21_vreg,    c_30_c_31_vreg,
    c_02_c_03_vreg,    c_12_c_13_vreg,    c_22_c_23_vreg,    c_32_c_33_vreg,
    b_p0_b_p1_vreg,
    b_p2_b_p3_vreg,
    a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg; 

  c_00_c_01_vreg.v = _mm_setzero_pd();   
  c_10_c_11_vreg.v = _mm_setzero_pd();
  c_20_c_21_vreg.v = _mm_setzero_pd(); 
  c_30_c_31_vreg.v = _mm_setzero_pd(); 
  c_02_c_03_vreg.v = _mm_setzero_pd();   
  c_12_c_13_vreg.v = _mm_setzero_pd();  
  c_22_c_23_vreg.v = _mm_setzero_pd();   
  c_32_c_33_vreg.v = _mm_setzero_pd(); 

  for (int p=0; p<k; p++ ){
    b_p0_b_p1_vreg.v = _mm_load_pd( (double *) b );
    b_p2_a_p3_vreg.v = _mm_load_pd( (double *) (b+2) );
    b+=4;

    a_0p_vreg.v = _mm_loaddup_pd( (double *) a );   /* load and duplicate */
    a_1p_vreg.v = _mm_loaddup_pd( (double *) (a+1));   /* load and duplicate */
    a_2p_vreg.v = _mm_loaddup_pd( (double *) (a+2) );   /* load and duplicate */
    a_3p_vreg.v = _mm_loaddup_pd( (double *) (a+3) );   /* load and duplicate */

    a+=4;
    /* First row and second rows */
    c_00_c_01_vreg.v += a_0p_vreg.v * b_p0_b_p1_vreg.v;
    c_10_c_11_vreg.v += a_1p_vreg.v * b_p0_b_p1_vreg.v;
    c_20_c_21_vreg.v += a_2p_vreg.v * b_p0_b_p1_vreg.v;
    c_30_c_31_vreg.v += a_3p_vreg.v * b_p0_b_p1_vreg.v;

    /* Third and fourth rows */
    c_02_c_03_vreg.v += a_0p_vreg.v * b_p2_b_p3_vreg.v;
    c_12_c_13_vreg.v += a_1p_vreg.v * b_p2_b_p3_vreg.v;
    c_22_c_23_vreg.v += a_2p_vreg.v * b_p2_b_p3_vreg.v;
    c_32_c_33_vreg.v += a_3p_vreg.v * b_p2_b_p3_vreg.v;
  }

  C( 0, 0 ) += c_00_c_01_vreg.d[0];  C( 1, 0 ) += c_10_c_11_vreg.d[0];  
  C( 2, 0 ) += c_20_c_21_vreg.d[0];  C( 3, 0 ) += c_30_c_31_vreg.d[0]; 

  C( 0, 1 ) += c_00_c_01_vreg.d[1];  C( 1, 1 ) += c_10_c_11_vreg.d[1];  
  C( 2, 1 ) += c_20_c_21_vreg.d[1];  C( 3, 1 ) += c_30_c_31_vreg.d[1]; 

  C( 0, 2 ) += c_02_c_03_vreg.d[0];  C( 1, 2 ) += c_12_c_13_vreg.d[0];  
  C( 2, 2 ) += c_22_c_23_vreg.d[0];  C( 3, 2 ) += c_32_c_33_vreg.d[0]; 

  C( 0, 3 ) += c_02_c_03_vreg.d[1];  C( 1, 3 ) += c_12_c_13_vreg.d[1];  
  C( 2, 3 ) += c_22_c_23_vreg.d[1];  C( 3, 3 ) += c_32_c_33_vreg.d[1]; 
}
