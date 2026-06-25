#include <stdlib.h>

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Block sizes */
#define nc 128
#define kc 256
#define ma 2048
#define NR 16

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */
void AddDot4x16( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
void InnerKernel( int m, int n, int k, double *a, int lda, 
    double *b, int ldb, double *c, int ldc, int first_time );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, int, double *, int, double * );

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
  static double *packedA = NULL; // ��ָ̬�룬��̬����
  static int max_packedA_size = 0;
  // ��̬����64�ֽڶ����packedB��ֱ����aligned_alloc������װ������

  // ��ʼ��packedA���״ε��û���Ҫ����ʱ��
  if (first_time) {
    int required_size = m * k * sizeof(double);
    if (packedA == NULL || required_size > max_packedA_size) {
      if (packedA != NULL) free(packedA); // �ͷž��ڴ�
      packedA = (double*)aligned_alloc(64, required_size);  // 64�ֽڶ���
      max_packedA_size = required_size;
    }
  }

  double *packedB = (double*)aligned_alloc(64, kc * nc * sizeof(double));
  for (int i=0; i<m; i+=4 ){ /* Loop over the rows of C */  
    if(first_time)
      PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
    int j;
    for (j=0; j + NR <= n; j+=NR ){ /* Loop over the columns of C*/
      /* Update C( i,j ), C( i,j+1 ), ..., C( i,j+NR-1 ) in
	 one routine */
      if(i==0)
        PackMatrixB( k, NR, &B( 0, j ), ldb, &packedB[ j*k ] );
      AddDot4x16( k, &packedA[ i*k ], k , &packedB[j*k], NR, &C( i,j ), ldc );
    }
    if (j < n) {
      int remaining = n - j;
      if(i==0)
      PackMatrixB( k, remaining, &B( 0, j ), ldb, &packedB[ j*k ] );
      for (int p=0; p<k; p++) {
        for (int ii=0; ii<4; ii++) {
          double a_val = packedA[i*k + p*4 + ii];
          for (int jj=0; jj<remaining; jj++) {
            C(i+ii, j+jj) += a_val * packedB[j*k + p*NR + jj];
          }
        }
      }
    }
  }
  free(packedB);  // �ͷ�packedB
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

void PackMatrixB( int k, int ncols, double *b, int ldb, double *b_to )
{
  for(int i=0; i<k; i++){  /* loop over rows of B */
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
