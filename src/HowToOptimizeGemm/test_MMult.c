#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 

#include "parameters.h"

void REF_MMult(int, int, int, double *, int, double *, int, double *, int );
void MY_MMult(int, int, int, double *, int, double *, int, double *, int );
void copy_matrix(int, int, double *, int, double *, int );
void random_matrix(int, int, double *, int);
double compare_matrices( int, int, double *, int, double *, int );
double dclock();

// 对齐内存分配函数（64字节对齐）
static double* aligned_malloc(size_t elements) {
    // AVX512需要64字节对齐，分配时按64字节边界对齐
    double* ptr = (double*)aligned_alloc(64, elements * sizeof(double));
    if (ptr == NULL) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }
    // 验证对齐（调试用，可删除）
    if (((uintptr_t)ptr % 64) != 0) {
        fprintf(stderr, "警告：内存未按64字节对齐\n");
    }
    return ptr;
}

int main()
{
  int p, m, n, k, lda, ldb, ldc, rep;
  double dtime, dtime_best, gflops, diff;
  double *a, *b, *c, *c_ref, *c_old;    
  
  printf( "MY_MMult = [\n" );    
  for ( p=PFIRST; p<=PLAST; p+=PINC ){
    m = ( M == -1 ? p : M );
    n = ( N == -1 ? p : N );
    k = ( K == -1 ? p : K );

    lda = ( LDA == -1 ? m : LDA );
    ldb = ( LDB == -1 ? k : LDB );
    ldc = ( LDC == -1 ? m : LDC );

        /* 分配对齐的矩阵内存 */
        /* 注意：A多分配一列，避免预取越界导致段错误 */
        a = aligned_malloc( lda * (k + 1) ); // 64字节对齐
        b = aligned_malloc( ldb * n );
        c = aligned_malloc( ldc * n );
        c_old = aligned_malloc( ldc * n );
        c_ref = aligned_malloc( ldc * n );

    /* Generate random matrices A, B, c_old */
    random_matrix( m, k, a, lda );
    random_matrix( k, n, b, ldb );
    random_matrix( m, n, c_old, ldc );

    copy_matrix( m, n, c_old, ldc, c_ref, ldc );

    /* Run the reference implementation so the answers can be compared */
    REF_MMult( m, n, k, a, lda, b, ldb, c_ref, ldc );

    /* Time the "optimized" implementation */
    for ( rep=0; rep<NREPEATS; rep++ ){
      copy_matrix( m, n, c_old, ldc, c, ldc );

      /* Time your implementation */
      dtime = dclock();
      MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );      
      dtime = dclock() - dtime;

      if ( rep==0 ){
        dtime_best = dtime;
      }else{
        dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
      }
    }
    gflops = 2.0 * m * n * k * 1.0e-09;
    diff = compare_matrices( m, n, c, ldc, c_ref, ldc );
    printf( "%d %le %le \n", p, gflops / dtime_best, diff );
    fflush( stdout );

    free( a );
    free( b );
    free( c );
    free( c_old );
    free( c_ref );
  }

  printf( "];\n" );
  exit( 0 );
}