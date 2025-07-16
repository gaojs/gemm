/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */
void AddDot4x4( int, double *, int, double *, int, double *, int );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int j=0; j<n; j+=4 ){ /* Loop over the rows of C */
    for (int i=0; i<m; i+=4 ){ /* Loop over the columns of C, unrolled by 4 */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	   one routine (four inner products) */
      AddDot4x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}


#include <immintrin.h>

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
    c_00_c_01_vreg, c_02_c_03_vreg,
    c_10_c_11_vreg, c_12_c_13_vreg,
    c_20_c_21_vreg, c_22_c_23_vreg,
    c_30_c_31_vreg, c_32_c_33_vreg,
    b_p0_b_p1_vreg, b_p2_a_p3_vreg,
    a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg; 

  double 
    /* Point to the current elements in the four columns of B */
    *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr; 
    
  a_0p_pntr = &A( 0, 0 );
  a_1p_pntr = &A( 1, 0 );
  a_2p_pntr = &A( 2, 0 );
  a_3p_pntr = &A( 3, 0 );

  c_00_c_01_vreg.v = _mm_setzero_pd();   
  c_10_c_11_vreg.v = _mm_setzero_pd();
  c_20_c_21_vreg.v = _mm_setzero_pd(); 
  c_30_c_31_vreg.v = _mm_setzero_pd(); 
  c_02_c_03_vreg.v = _mm_setzero_pd();
  c_12_c_13_vreg.v = _mm_setzero_pd();  
  c_22_c_23_vreg.v = _mm_setzero_pd();   
  c_32_c_33_vreg.v = _mm_setzero_pd(); 

  for (int p=0; p<k; p++ ){
    b_p0_b_p1_vreg.v = _mm_load_pd( (double *) &B( p, 0 ) );
    b_p2_a_p3_vreg.v = _mm_load_pd( (double *) &B( p, 2 ) );

    a_0p_vreg.v = _mm_loaddup_pd( (double *) a_0p_pntr++ );   /* load and duplicate */
    a_1p_vreg.v = _mm_loaddup_pd( (double *) a_1p_pntr++ );   /* load and duplicate */
    a_2p_vreg.v = _mm_loaddup_pd( (double *) a_2p_pntr++ );   /* load and duplicate */
    a_3p_vreg.v = _mm_loaddup_pd( (double *) a_3p_pntr++ );   /* load and duplicate */

    /* First row and second rows */
    c_00_c_01_vreg.v += a_0p_vreg.v * b_p0_b_p1_vreg.v;
    c_10_c_11_vreg.v += a_1p_vreg.v * b_p0_b_p1_vreg.v;
    c_20_c_21_vreg.v += a_2p_vreg.v * b_p0_b_p1_vreg.v;
    c_30_c_31_vreg.v += a_3p_vreg.v * b_p0_b_p1_vreg.v;

    /* Third and fourth rows */
    c_02_c_03_vreg.v += a_0p_vreg.v * b_p2_a_p3_vreg.v;
    c_12_c_13_vreg.v += a_1p_vreg.v * b_p2_a_p3_vreg.v;
    c_22_c_23_vreg.v += a_2p_vreg.v * b_p2_a_p3_vreg.v;
    c_32_c_33_vreg.v += a_3p_vreg.v * b_p2_a_p3_vreg.v;
  }

  C( 0, 0 ) += c_00_c_01_vreg.d[0];  C( 0, 1 ) += c_00_c_01_vreg.d[1];  
  C( 0, 2 ) += c_02_c_03_vreg.d[0];  C( 0, 3 ) += c_02_c_03_vreg.d[1];
  C( 1, 0 ) += c_10_c_11_vreg.d[0];  C( 1, 1 ) += c_10_c_11_vreg.d[1];  
  C( 1, 2 ) += c_12_c_13_vreg.d[0];  C( 1, 3 ) += c_12_c_13_vreg.d[1]; 
  C( 2, 0 ) += c_20_c_21_vreg.d[0];  C( 2, 1 ) += c_20_c_21_vreg.d[1];
  C( 2, 2 ) += c_22_c_23_vreg.d[0];  C( 2, 3 ) += c_22_c_23_vreg.d[1];
  C( 3, 0 ) += c_30_c_31_vreg.d[0];  C( 3, 1 ) += c_30_c_31_vreg.d[1];
  C( 3, 2 ) += c_32_c_33_vreg.d[0];  C( 3, 3 ) += c_32_c_33_vreg.d[1]; 
}
