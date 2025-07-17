#include <stdlib.h>

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
void AddDot4x16( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
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
  static double *packedA = NULL; // 静态指针，动态分配
  static int max_packedA_size = 0;
  // 动态分配64字节对齐的packedB（直接用aligned_alloc，不封装函数）

  // 初始化packedA（首次调用或需要扩容时）
  if (first_time) {
    int required_size = m * k * sizeof(double);
    if (packedA == NULL || required_size > max_packedA_size) {
      if (packedA != NULL) free(packedA); // 释放旧内存
      packedA = (double*)aligned_alloc(64, required_size);  // 64字节对齐
      max_packedA_size = required_size;
    }
  }

  double *packedB = (double*)aligned_alloc(64, k * n * sizeof(double));
  for (int i=0; i<m; i+=4 ){ /* Loop over the rows of C */  
    if(first_time)
      PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
    for (int j=0; j<n; j+=16 ){ /* Loop over the columns of C*/
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	   one routine (four inner products) */
      if(i==0)
        PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
      AddDot4x16( k, &packedA[ i*k ], k , &packedB[j*k], 16, &C( i,j ), ldc );
    }
  }
  free(packedB);  // 释放packedB
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
    *(b_to+8) = *(b_ji_pntr+8);
    *(b_to+9) = *(b_ji_pntr+9);
    *(b_to+10) = *(b_ji_pntr+10);
    *(b_to+11) = *(b_ji_pntr+11);
    *(b_to+12) = *(b_ji_pntr+12);
    *(b_to+13) = *(b_ji_pntr+13);
    *(b_to+14) = *(b_ji_pntr+14);
    *(b_to+15) = *(b_ji_pntr+15);
    b_to += 16;
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


void AddDot4x16( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{

  __m512d c0_0 = _mm512_setzero_pd();   
  __m512d c1_0 = _mm512_setzero_pd();
  __m512d c2_0 = _mm512_setzero_pd(); 
  __m512d c3_0 = _mm512_setzero_pd();
  __m512d c0_8 = _mm512_setzero_pd();   
  __m512d c1_8 = _mm512_setzero_pd();
  __m512d c2_8 = _mm512_setzero_pd(); 
  __m512d c3_8 = _mm512_setzero_pd(); 

  for (int p=0; p<k; p++ ){
    __m512d b_vec_0 = _mm512_load_pd( b );
    __m512d b_vec_8 = _mm512_load_pd( b+8 );
    b+=16;

    __m512d a0 = _mm512_broadcastsd_pd(_mm_load_sd(a) );   /* load and duplicate */
    __m512d a1 = _mm512_broadcastsd_pd(_mm_load_sd(a+1) );   /* load and duplicate */
    __m512d a2 = _mm512_broadcastsd_pd(_mm_load_sd(a+2) );   /* load and duplicate */
    __m512d a3 = _mm512_broadcastsd_pd(_mm_load_sd(a+3) );   /* load and duplicate */
    a+=4;

    c0_0 = _mm512_fmadd_pd(b_vec_0, a0, c0_0);
    c1_0 = _mm512_fmadd_pd(b_vec_0, a1, c1_0);
    c2_0 = _mm512_fmadd_pd(b_vec_0, a2, c2_0);
    c3_0 = _mm512_fmadd_pd(b_vec_0, a3, c3_0);
    c0_8 = _mm512_fmadd_pd(b_vec_8, a0, c0_8);
    c1_8 = _mm512_fmadd_pd(b_vec_8, a1, c1_8);
    c2_8 = _mm512_fmadd_pd(b_vec_8, a2, c2_8);
    c3_8 = _mm512_fmadd_pd(b_vec_8, a3, c3_8);

  }
  _mm512_storeu_pd(&C(0,0), _mm512_add_pd(_mm512_loadu_pd(&C(0,0)), c0_0));
  _mm512_storeu_pd(&C(1,0), _mm512_add_pd(_mm512_loadu_pd(&C(1,0)), c1_0));
  _mm512_storeu_pd(&C(2,0), _mm512_add_pd(_mm512_loadu_pd(&C(2,0)), c2_0));
  _mm512_storeu_pd(&C(3,0), _mm512_add_pd(_mm512_loadu_pd(&C(3,0)), c3_0));
  _mm512_storeu_pd(&C(0,8), _mm512_add_pd(_mm512_loadu_pd(&C(0,8)), c0_8));
  _mm512_storeu_pd(&C(1,8), _mm512_add_pd(_mm512_loadu_pd(&C(1,8)), c1_8));
  _mm512_storeu_pd(&C(2,8), _mm512_add_pd(_mm512_loadu_pd(&C(2,8)), c2_8));
  _mm512_storeu_pd(&C(3,8), _mm512_add_pd(_mm512_loadu_pd(&C(3,8)), c3_8));
}
