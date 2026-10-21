#include <stdio.h>

#define A( i, j ) a[ (i)*lda + (j) ]

void print_matrix( int m, int n, double *a, int lda )
{
  for ( int i=0; i<m; i++ ) {
    for ( int j=0; j<n; j++ ) {
      // l：length 长度修饰符，代表处理 double 双精度浮点数
      // e：exponent，科学计数法（指数形式），输出小写 e
      // 合起来：% le = 用小写 e 的科学计数法打印 double
      printf("%le ", A( i,j ) ); // 输出：5.000000e-01
    }
    printf("\n");
  }
  printf("\n");
}

