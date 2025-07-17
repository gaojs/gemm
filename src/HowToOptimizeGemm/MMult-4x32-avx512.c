
/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Block sizes */
#define nc 128
#define kc 512
#define ma 2048

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

void AddDot4x32( int, double *, int, double *, int, double *, int );
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
    for ( j=0; j<n; j+=32 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if( i == 0 )
        PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
      AddDot4x32( k, &packedA[ i*k ], k , &packedB[ j*k ], 32, &C( i,j ), ldc );
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
  int i,p;

  for( i=0; i<k; i++){  /* loop over rows of B */
    double 
      *b_ji_pntr = &B( i , 0 );
    
    for( p=0; p<32; p++){
        *(b_to + p) = *(b_ji_pntr + p);
    }
    b_to += 32;
  }
}

#include <immintrin.h>

typedef union
{
  __m512d v;
  double d[8];
} v8df_t;

void AddDot4x32( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
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


  v8df_t
    c0000_0007, c0008_0015, c0016_0023, c0024_0031,
    c0100_0107, c0108_0115, c0116_0123, c0124_0131,
    c0200_0207, c0208_0215, c0216_0223, c0224_0231,
    c0300_0307, c0308_0315, c0316_0323, c0324_0331, 
    b0x_7x, b8x_15x, b16x_23x, b24x_31x,
	ax0, ax1, ax2, ax3;


  c0000_0007.v = _mm512_setzero_pd();   
  c0008_0015.v = _mm512_setzero_pd();
  c0016_0023.v = _mm512_setzero_pd(); 
  c0024_0031.v = _mm512_setzero_pd(); 

  c0100_0107.v = _mm512_setzero_pd();   
  c0108_0115.v = _mm512_setzero_pd();
  c0116_0123.v = _mm512_setzero_pd(); 
  c0124_0131.v = _mm512_setzero_pd(); 

  c0200_0207.v = _mm512_setzero_pd();   
  c0208_0215.v = _mm512_setzero_pd();
  c0216_0223.v = _mm512_setzero_pd(); 
  c0224_0231.v = _mm512_setzero_pd(); 

  c0300_0307.v = _mm512_setzero_pd();   
  c0308_0315.v = _mm512_setzero_pd();
  c0316_0323.v = _mm512_setzero_pd(); 
  c0324_0331.v = _mm512_setzero_pd(); 

  int p;
  for ( p=0; p<k; p++ ){
    b0x_7x.v = _mm512_loadu_pd( (double *) b );
    b8x_15x.v = _mm512_loadu_pd( (double *) (b+8) );
    b16x_23x.v = _mm512_loadu_pd( (double *) (b+16) );
    b24x_31x.v = _mm512_loadu_pd( (double *) (b+24) );
    b+=32;

	  ax0.v = _mm512_broadcastsd_pd(_mm_load_sd(a));
	  ax1.v = _mm512_broadcastsd_pd(_mm_load_sd(a+1));
	  ax2.v = _mm512_broadcastsd_pd(_mm_load_sd(a+2));
	  ax3.v = _mm512_broadcastsd_pd(_mm_load_sd(a+3));

    a+=4;

    // c0000_0007.v += ax0.v * b0x_7x.v;
    // c0100_0107.v += ax1.v * b0x_7x.v;
    // c0200_0207.v += ax2.v * b0x_7x.v;
    // c0300_0307.v += ax3.v * b0x_7x.v;    

    c0000_0007.v = _mm512_add_pd(_mm512_mul_pd(ax0.v, b0x_7x.v), c0000_0007.v);
    c0100_0107.v = _mm512_add_pd(_mm512_mul_pd(ax1.v, b0x_7x.v), c0100_0107.v);
    c0200_0207.v = _mm512_add_pd(_mm512_mul_pd(ax2.v, b0x_7x.v), c0200_0207.v);
    c0300_0307.v = _mm512_add_pd(_mm512_mul_pd(ax3.v, b0x_7x.v), c0300_0307.v);

    // c0008_0015.v += ax0.v * b8x_15x.v;
    // c0108_0115.v += ax1.v * b8x_15x.v;
    // c0208_0215.v += ax2.v * b8x_15x.v;
    // c0308_0315.v += ax3.v * b8x_15x.v;   

    c0008_0015.v = _mm512_add_pd(_mm512_mul_pd(ax0.v, b8x_15x.v), c0008_0015.v);
    c0108_0115.v = _mm512_add_pd(_mm512_mul_pd(ax1.v, b8x_15x.v), c0108_0115.v);
    c0208_0215.v = _mm512_add_pd(_mm512_mul_pd(ax2.v, b8x_15x.v), c0208_0215.v);
    c0308_0315.v = _mm512_add_pd(_mm512_mul_pd(ax3.v, b8x_15x.v), c0308_0315.v);


    // c0016_0023.v += ax0.v * b16x_23x.v;
    // c0116_0123.v += ax1.v * b16x_23x.v;
    // c0216_0223.v += ax2.v * b16x_23x.v;
    // c0316_0323.v += ax3.v * b16x_23x.v;   

    c0016_0023.v = _mm512_add_pd(_mm512_mul_pd(ax0.v, b16x_23x.v), c0016_0023.v);
    c0116_0123.v = _mm512_add_pd(_mm512_mul_pd(ax1.v, b16x_23x.v), c0116_0123.v);
    c0216_0223.v = _mm512_add_pd(_mm512_mul_pd(ax2.v, b16x_23x.v), c0216_0223.v);
    c0316_0323.v = _mm512_add_pd(_mm512_mul_pd(ax3.v, b16x_23x.v), c0316_0323.v);

    // c0024_0031.v += ax0.v * b24x_31x.v;
    // c0124_0131.v += ax1.v * b24x_31x.v;
    // c0224_0231.v += ax2.v * b24x_31x.v;
    // c0324_0331.v += ax3.v * b24x_31x.v;       

    c0024_0031.v = _mm512_add_pd(_mm512_mul_pd(ax0.v, b24x_31x.v), c0024_0031.v);
    c0124_0131.v = _mm512_add_pd(_mm512_mul_pd(ax1.v, b24x_31x.v), c0124_0131.v);
    c0224_0231.v = _mm512_add_pd(_mm512_mul_pd(ax2.v, b24x_31x.v), c0224_0231.v);
    c0324_0331.v = _mm512_add_pd(_mm512_mul_pd(ax3.v, b24x_31x.v), c0324_0331.v);    

  }


  C( 0, 0 ) += c0000_0007.d[0];  C( 0, 1 ) += c0000_0007.d[1];  
  C( 0, 2 ) += c0000_0007.d[2];  C( 0, 3 ) += c0000_0007.d[3];  
  C( 0, 4 ) += c0000_0007.d[4];  C( 0, 5 ) += c0000_0007.d[5];  
  C( 0, 6 ) += c0000_0007.d[6];  C( 0, 7 ) += c0000_0007.d[7];  
  C( 0, 8 ) += c0008_0015.d[0];  C( 0, 9 ) += c0008_0015.d[1];  
  C( 0, 10 ) += c0008_0015.d[2];  C( 0, 11 ) += c0008_0015.d[3];  
  C( 0, 12 ) += c0008_0015.d[4];  C( 0, 13 ) += c0008_0015.d[5];  
  C( 0, 14 ) += c0008_0015.d[6];  C( 0, 15 ) += c0008_0015.d[7];
  C( 0, 16 ) += c0016_0023.d[0];  C( 0, 17 ) += c0016_0023.d[1];  
  C( 0, 18 ) += c0016_0023.d[2];  C( 0, 19 ) += c0016_0023.d[3];  
  C( 0, 20 ) += c0016_0023.d[4];  C( 0, 21 ) += c0016_0023.d[5];  
  C( 0, 22 ) += c0016_0023.d[6];  C( 0, 23 ) += c0016_0023.d[7];  
  C( 0, 24 ) += c0024_0031.d[0];  C( 0, 25 ) += c0024_0031.d[1];  
  C( 0, 26 ) += c0024_0031.d[2];  C( 0, 27 ) += c0024_0031.d[3];  
  C( 0, 28 ) += c0024_0031.d[4];  C( 0, 29 ) += c0024_0031.d[5];  
  C( 0, 30 ) += c0024_0031.d[6];  C( 0, 31 ) += c0024_0031.d[7];  


  C( 1, 0 ) += c0100_0107.d[0];  C( 1, 1 ) += c0100_0107.d[1];  
  C( 1, 2 ) += c0100_0107.d[2];  C( 1, 3 ) += c0100_0107.d[3];  
  C( 1, 4 ) += c0100_0107.d[4];  C( 1, 5 ) += c0100_0107.d[5];  
  C( 1, 6 ) += c0100_0107.d[6];  C( 1, 7 ) += c0100_0107.d[7];  
  C( 1, 8 ) += c0108_0115.d[0];  C( 1, 9 ) += c0108_0115.d[1];  
  C( 1, 10 ) += c0108_0115.d[2];  C( 1, 11 ) += c0108_0115.d[3];  
  C( 1, 12 ) += c0108_0115.d[4];  C( 1, 13 ) += c0108_0115.d[5];  
  C( 1, 14 ) += c0108_0115.d[6];  C( 1, 15 ) += c0108_0115.d[7];
  C( 1, 16 ) += c0116_0123.d[0];  C( 1, 17 ) += c0116_0123.d[1];  
  C( 1, 18 ) += c0116_0123.d[2];  C( 1, 19 ) += c0116_0123.d[3];  
  C( 1, 20 ) += c0116_0123.d[4];  C( 1, 21 ) += c0116_0123.d[5];  
  C( 1, 22 ) += c0116_0123.d[6];  C( 1, 23 ) += c0116_0123.d[7];  
  C( 1, 24 ) += c0124_0131.d[0];  C( 1, 25 ) += c0124_0131.d[1];  
  C( 1, 26 ) += c0124_0131.d[2];  C( 1, 27 ) += c0124_0131.d[3];  
  C( 1, 28 ) += c0124_0131.d[4];  C( 1, 29 ) += c0124_0131.d[5];  
  C( 1, 30 ) += c0124_0131.d[6];  C( 1, 31 ) += c0124_0131.d[7];  

  C( 2, 0 ) += c0200_0207.d[0];  C( 2, 1 ) += c0200_0207.d[1];  
  C( 2, 2 ) += c0200_0207.d[2];  C( 2, 3 ) += c0200_0207.d[3];  
  C( 2, 4 ) += c0200_0207.d[4];  C( 2, 5 ) += c0200_0207.d[5];  
  C( 2, 6 ) += c0200_0207.d[6];  C( 2, 7 ) += c0200_0207.d[7];  
  C( 2, 8 ) += c0208_0215.d[0];  C( 2, 9 ) += c0208_0215.d[1];  
  C( 2, 10 ) += c0208_0215.d[2];  C( 2, 11 ) += c0208_0215.d[3];  
  C( 2, 12 ) += c0208_0215.d[4];  C( 2, 13 ) += c0208_0215.d[5];  
  C( 2, 14 ) += c0208_0215.d[6];  C( 2, 15 ) += c0208_0215.d[7];
  C( 2, 16 ) += c0216_0223.d[0];  C( 2, 17 ) += c0216_0223.d[1];  
  C( 2, 18 ) += c0216_0223.d[2];  C( 2, 19 ) += c0216_0223.d[3];  
  C( 2, 20 ) += c0216_0223.d[4];  C( 2, 21 ) += c0216_0223.d[5];  
  C( 2, 22 ) += c0216_0223.d[6];  C( 2, 23 ) += c0216_0223.d[7];  
  C( 2, 24 ) += c0224_0231.d[0];  C( 2, 25 ) += c0224_0231.d[1];  
  C( 2, 26 ) += c0224_0231.d[2];  C( 2, 27 ) += c0224_0231.d[3];  
  C( 2, 28 ) += c0224_0231.d[4];  C( 2, 29 ) += c0224_0231.d[5];  
  C( 2, 30 ) += c0224_0231.d[6];  C( 2, 31 ) += c0224_0231.d[7];  


  C( 3, 0 ) += c0300_0307.d[0];  C( 3, 1 ) += c0300_0307.d[1];  
  C( 3, 2 ) += c0300_0307.d[2];  C( 3, 3 ) += c0300_0307.d[3];  
  C( 3, 4 ) += c0300_0307.d[4];  C( 3, 5 ) += c0300_0307.d[5];  
  C( 3, 6 ) += c0300_0307.d[6];  C( 3, 7 ) += c0300_0307.d[7];  
  C( 3, 8 ) += c0308_0315.d[0];  C( 3, 9 ) += c0308_0315.d[1];  
  C( 3, 10 ) += c0308_0315.d[2];  C( 3, 11 ) += c0308_0315.d[3];
  C( 3, 12 ) += c0308_0315.d[4];  C( 3, 13 ) += c0308_0315.d[5];  
  C( 3, 14 ) += c0308_0315.d[6];  C( 3, 15 ) += c0308_0315.d[7];
  C( 3, 16 ) += c0316_0323.d[0];  C( 3, 17 ) += c0316_0323.d[1];  
  C( 3, 18 ) += c0316_0323.d[2];  C( 3, 19 ) += c0316_0323.d[3];  
  C( 3, 20 ) += c0316_0323.d[4];  C( 3, 21 ) += c0316_0323.d[5];  
  C( 3, 22 ) += c0316_0323.d[6];  C( 3, 23 ) += c0316_0323.d[7];  
  C( 3, 24 ) += c0324_0331.d[0];  C( 3, 25 ) += c0324_0331.d[1];  
  C( 3, 26 ) += c0324_0331.d[2];  C( 3, 27 ) += c0324_0331.d[3];  
  C( 3, 28 ) += c0324_0331.d[4];  C( 3, 29 ) += c0324_0331.d[5];  
  C( 3, 30 ) += c0324_0331.d[6];  C( 3, 31 ) += c0324_0331.d[7];  

}
