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
void AddDot8x4( int, double *, int, double *, int, double *, int );
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
    for (int j=0; j<n; j+=8 ){ /* Loop over the columns of C*/
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if(i==0)
      PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
      AddDot8x4( k, &packedA[ i*k ], k , &packedB[j*k], 8, &C( i,j ), ldc );
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
    *(b_to+4) = *(b_ji_pntr+4);
    *(b_to+5) = *(b_ji_pntr+5);
    *(b_to+6) = *(b_ji_pntr+6);
    *(b_to+7) = *(b_ji_pntr+7);
    b_to += 8;
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


void AddDot8x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{

  __m512d c0 = _mm512_setzero_pd();   
  __m512d c1 = _mm512_setzero_pd();
  __m512d c2 = _mm512_setzero_pd(); 
  __m512d c3 = _mm512_setzero_pd(); 

  for (int p=0; p<k; p++ ){
    __m512d b_vec = _mm512_loadu_pd( b );
    b+=8;

    __m512d a0 = _mm512_broadcastsd_pd(_mm_load_sd(a) );   /* load and duplicate */
    __m512d a1 = _mm512_broadcastsd_pd(_mm_load_sd(a+1) );   /* load and duplicate */
    __m512d a2 = _mm512_broadcastsd_pd(_mm_load_sd(a+2) );   /* load and duplicate */
    __m512d a3 = _mm512_broadcastsd_pd(_mm_load_sd(a+3) );   /* load and duplicate */
    a+=4;

    c0 = _mm512_fmadd_pd(b_vec, a0, c0);
    c1 = _mm512_fmadd_pd(b_vec, a1, c1);
    c2 = _mm512_fmadd_pd(b_vec, a2, c2);
    c3 = _mm512_fmadd_pd(b_vec, a3, c3);

  }
  _mm512_storeu_pd(&C(0,0), _mm512_add_pd(_mm512_loadu_pd(&C(0,0)), c0));
  _mm512_storeu_pd(&C(1,0), _mm512_add_pd(_mm512_loadu_pd(&C(1,0)), c1));
  _mm512_storeu_pd(&C(2,0), _mm512_add_pd(_mm512_loadu_pd(&C(2,0)), c2));
  _mm512_storeu_pd(&C(3,0), _mm512_add_pd(_mm512_loadu_pd(&C(3,0)), c3));
}
