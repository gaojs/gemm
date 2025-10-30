#include <immintrin.h>

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

void dot4x4_avx256(int k, double *a, int lda,
    double *b, int ldb,double *c, int ldc)
{
  __m256d mc0 = _mm256_loadu_pd(&C(0, 0));
  __m256d mc1 = _mm256_loadu_pd(&C(1, 0));
  __m256d mc2 = _mm256_loadu_pd(&C(2, 0));
  __m256d mc3 = _mm256_loadu_pd(&C(3, 0));
  for (int p=0; p<k; p++ ){
    __m256d ma0 = _mm256_set1_pd(A(0,p));
    __m256d ma1 = _mm256_set1_pd(A(1,p));
    __m256d ma2 = _mm256_set1_pd(A(2,p));
    __m256d ma3 = _mm256_set1_pd(A(3,p));
    __m256d mb0 = _mm256_loadu_pd(&B(p, 0));
    mc0 = _mm256_fmadd_pd(ma0, mb0, mc0);
    mc1 = _mm256_fmadd_pd(ma1, mb0, mc1);
    mc2 = _mm256_fmadd_pd(ma2, mb0, mc2);
    mc3 = _mm256_fmadd_pd(ma3, mb0, mc3);
  }  
  _mm256_storeu_pd(&C(0, 0), mc0);
  _mm256_storeu_pd(&C(1, 0), mc1);
  _mm256_storeu_pd(&C(2, 0), mc2);
  _mm256_storeu_pd(&C(3, 0), mc3);
}

void dot4x8_avx256(int k, double *a, int lda,
    double *b, int ldb, double *c, int ldc) {
    // 加载C的4行×8列初始值（每行8个元素，用2个AVX2向量存储）
    __m256d c0_0 = _mm256_loadu_pd(&C(0, 0));  // C(0,0-3)
    __m256d c0_1 = _mm256_loadu_pd(&C(0, 4));  // C(0,4-7)
    __m256d c1_0 = _mm256_loadu_pd(&C(1, 0));  // C(1,0-3)
    __m256d c1_1 = _mm256_loadu_pd(&C(1, 4));  // C(1,4-7)
    __m256d c2_0 = _mm256_loadu_pd(&C(2, 0));  // C(2,0-3)
    __m256d c2_1 = _mm256_loadu_pd(&C(2, 4));  // C(2,4-7)
    __m256d c3_0 = _mm256_loadu_pd(&C(3, 0));  // C(3,0-3)
    __m256d c3_1 = _mm256_loadu_pd(&C(3, 4));  // C(3,4-7)
    for (int p = 0; p < k; p++) {
        // 广播A的4行当前列元素（A(0,p), A(1,p), A(2,p), A(3,p)）
        __m256d a0 = _mm256_set1_pd(A(0, p));
        __m256d a1 = _mm256_set1_pd(A(1, p));
        __m256d a2 = _mm256_set1_pd(A(2, p));
        __m256d a3 = _mm256_set1_pd(A(3, p));
        // 加载B当前行的8列（分2个向量）
        __m256d b0 = _mm256_loadu_pd(&B(p, 0));   // B(p,0-3)
        __m256d b1 = _mm256_loadu_pd(&B(p, 4));   // B(p,4-7)
        // 融合乘减：C -= A*B（分别处理前4列和后4列）
        c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
        c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);
        c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
        c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);
        c2_0 = _mm256_fmadd_pd(a2, b0, c2_0);
        c2_1 = _mm256_fmadd_pd(a2, b1, c2_1);
        c3_0 = _mm256_fmadd_pd(a3, b0, c3_0);
        c3_1 = _mm256_fmadd_pd(a3, b1, c3_1);
    }
    // 存储结果回C
    _mm256_storeu_pd(&C(0, 0), c0_0);
    _mm256_storeu_pd(&C(0, 4), c0_1);
    _mm256_storeu_pd(&C(1, 0), c1_0);
    _mm256_storeu_pd(&C(1, 4), c1_1);
    _mm256_storeu_pd(&C(2, 0), c2_0);
    _mm256_storeu_pd(&C(2, 4), c2_1);
    _mm256_storeu_pd(&C(3, 0), c3_0);
    _mm256_storeu_pd(&C(3, 4), c3_1);
}

void dot8x16_avx512(int k, double *a, int lda,
    double *b, int ldb, double *c, int ldc) {
    // 加载C的初始值（8行×16列，每行2个AVX512向量）
    __m512d c0[2] = {_mm512_loadu_pd(&C(0, 0)), _mm512_loadu_pd(&C(0, 8))};
    __m512d c1[2] = {_mm512_loadu_pd(&C(1, 0)), _mm512_loadu_pd(&C(1, 8))};
    __m512d c2[2] = {_mm512_loadu_pd(&C(2, 0)), _mm512_loadu_pd(&C(2, 8))};
    __m512d c3[2] = {_mm512_loadu_pd(&C(3, 0)), _mm512_loadu_pd(&C(3, 8))};
    __m512d c4[2] = {_mm512_loadu_pd(&C(4, 0)), _mm512_loadu_pd(&C(4, 8))};
    __m512d c5[2] = {_mm512_loadu_pd(&C(5, 0)), _mm512_loadu_pd(&C(5, 8))};
    __m512d c6[2] = {_mm512_loadu_pd(&C(6, 0)), _mm512_loadu_pd(&C(6, 8))};
    __m512d c7[2] = {_mm512_loadu_pd(&C(7, 0)), _mm512_loadu_pd(&C(7, 8))};

    for (int p = 0; p < k; p++) {
        // 广播A的8行当前列元素
        __m512d a0 = _mm512_set1_pd(A(0, p));
        __m512d a1 = _mm512_set1_pd(A(1, p));
        __m512d a2 = _mm512_set1_pd(A(2, p));
        __m512d a3 = _mm512_set1_pd(A(3, p));
        __m512d a4 = _mm512_set1_pd(A(4, p));
        __m512d a5 = _mm512_set1_pd(A(5, p));
        __m512d a6 = _mm512_set1_pd(A(6, p));
        __m512d a7 = _mm512_set1_pd(A(7, p));

        // 加载B当前行的16列（2个AVX512向量）
        __m512d b0 = _mm512_loadu_pd(&B(p, 0));  // B[p][0-7]
        __m512d b1 = _mm512_loadu_pd(&B(p, 8));  // B[p][8-15]

        // 融合乘减：并行处理16列
        c0[0] = _mm512_fmadd_pd(a0, b0, c0[0]);
        c0[1] = _mm512_fmadd_pd(a0, b1, c0[1]);
        c1[0] = _mm512_fmadd_pd(a1, b0, c1[0]);
        c1[1] = _mm512_fmadd_pd(a1, b1, c1[1]);
        c2[0] = _mm512_fmadd_pd(a2, b0, c2[0]);
        c2[1] = _mm512_fmadd_pd(a2, b1, c2[1]);
        c3[0] = _mm512_fmadd_pd(a3, b0, c3[0]);
        c3[1] = _mm512_fmadd_pd(a3, b1, c3[1]);
        c4[0] = _mm512_fmadd_pd(a4, b0, c4[0]);
        c4[1] = _mm512_fmadd_pd(a4, b1, c4[1]);
        c5[0] = _mm512_fmadd_pd(a5, b0, c5[0]);
        c5[1] = _mm512_fmadd_pd(a5, b1, c5[1]);
        c6[0] = _mm512_fmadd_pd(a6, b0, c6[0]);
        c6[1] = _mm512_fmadd_pd(a6, b1, c6[1]);
        c7[0] = _mm512_fmadd_pd(a7, b0, c7[0]);
        c7[1] = _mm512_fmadd_pd(a7, b1, c7[1]);
    }

    // 存储结果回C
    _mm512_storeu_pd(&C(0, 0), c0[0]);
    _mm512_storeu_pd(&C(0, 8), c0[1]);
    _mm512_storeu_pd(&C(1, 0), c1[0]);
    _mm512_storeu_pd(&C(1, 8), c1[1]);
    _mm512_storeu_pd(&C(2, 0), c2[0]);
    _mm512_storeu_pd(&C(2, 8), c2[1]);
    _mm512_storeu_pd(&C(3, 0), c3[0]);
    _mm512_storeu_pd(&C(3, 8), c3[1]);
    _mm512_storeu_pd(&C(4, 0), c4[0]);
    _mm512_storeu_pd(&C(4, 8), c4[1]);
    _mm512_storeu_pd(&C(5, 0), c5[0]);
    _mm512_storeu_pd(&C(5, 8), c5[1]);
    _mm512_storeu_pd(&C(6, 0), c6[0]);
    _mm512_storeu_pd(&C(6, 8), c6[1]);
    _mm512_storeu_pd(&C(7, 0), c7[0]);
    _mm512_storeu_pd(&C(7, 8), c7[1]);
}

// AVX512 8x8 微内核：计算 C(8x8) -= A(8xk) * B(kx8)
// 8行×8列，作为8x16的补充
// 8行×1向量 = 8个C向量 + 8个A广播 + 1个B加载 = 17个寄存器（≤32）
void dot8x8_avx512(int k, double *a, int lda,
    double *b, int ldb, double *c, int ldc) {
    // 加载C的初始值（8行×8列，每行1个AVX512向量）
    __m512d c0 = _mm512_loadu_pd(&C(0, 0));
    __m512d c1 = _mm512_loadu_pd(&C(1, 0));
    __m512d c2 = _mm512_loadu_pd(&C(2, 0));
    __m512d c3 = _mm512_loadu_pd(&C(3, 0));
    __m512d c4 = _mm512_loadu_pd(&C(4, 0));
    __m512d c5 = _mm512_loadu_pd(&C(5, 0));
    __m512d c6 = _mm512_loadu_pd(&C(6, 0));
    __m512d c7 = _mm512_loadu_pd(&C(7, 0));

    for (int p = 0; p < k; p++) {
        // 广播A的8行当前列元素
        __m512d a0 = _mm512_set1_pd(A(0, p));
        __m512d a1 = _mm512_set1_pd(A(1, p));
        __m512d a2 = _mm512_set1_pd(A(2, p));
        __m512d a3 = _mm512_set1_pd(A(3, p));
        __m512d a4 = _mm512_set1_pd(A(4, p));
        __m512d a5 = _mm512_set1_pd(A(5, p));
        __m512d a6 = _mm512_set1_pd(A(6, p));
        __m512d a7 = _mm512_set1_pd(A(7, p));

        // 加载B当前行的8列
        __m512d b0 = _mm512_loadu_pd(&B(p, 0));  // B[p][0-7]

        // 融合乘减：并行处理8列
        c0 = _mm512_fmadd_pd(a0, b0, c0);
        c1 = _mm512_fmadd_pd(a1, b0, c1);
        c2 = _mm512_fmadd_pd(a2, b0, c2);
        c3 = _mm512_fmadd_pd(a3, b0, c3);
        c4 = _mm512_fmadd_pd(a4, b0, c4);
        c5 = _mm512_fmadd_pd(a5, b0, c5);
        c6 = _mm512_fmadd_pd(a6, b0, c6);
        c7 = _mm512_fmadd_pd(a7, b0, c7);
    }

    // 存储结果回C
    _mm512_storeu_pd(&C(0, 0), c0);
    _mm512_storeu_pd(&C(1, 0), c1);
    _mm512_storeu_pd(&C(2, 0), c2);
    _mm512_storeu_pd(&C(3, 0), c3);
    _mm512_storeu_pd(&C(4, 0), c4);
    _mm512_storeu_pd(&C(5, 0), c5);
    _mm512_storeu_pd(&C(6, 0), c6);
    _mm512_storeu_pd(&C(7, 0), c7);
}

void kernel( int m, int n, int k, double *a, int lda, 
  double *b, int ldb, double *c, int ldc)
{
  for (int i=0; i<m; i+=8){
    for (int j=0; j<n; j+=8){
      dot8x8_avx512(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
    }
  }  
}

/* Block sizes */
#define kc 48
#define nc 32
#define mc 32

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
  double *b, int ldb,double *c, int ldc )
{
  for (int i=0; i<m; i+=mc){
    int mb = min( m-i, mc);
    for (int p=0; p<k; p+=kc){
      int kb = min( k-p, kc);
      for (int j=0; j<n; j+=nc){
        int nb = min( n-j, nc);
        kernel(mb, nb, kb, &A(i, p), lda, &B(p, j), ldb, &C(i,j), ldc);
      }
    }
  }
}
 
