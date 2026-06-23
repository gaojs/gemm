#include <stdlib.h>

// lda = leading dimension，矩阵实际分配行数
#define A( i,j ) a[ (i)*lda + (j) ]

void random_matrix( int m, int n, double *a, int lda )
{
  // srand48(long seed)：设置随机种子，不设置则默认种子固定，每次运行矩阵完全一样
  // lrand48()：返回非负 long 整数；mrand48()：返回有符号 long，范围 [-2^31, 2^31)
  // drand48() 是 POSIX 标准的伪随机浮点数生成函数，定义在 <stdlib.h>
  // Linux/macOS 原生支持，Windows MSVC 没有这个函数。
  // drand48 使用全局静态内部状态，多线程同时调用会产生数据竞争，随机序列错乱。
  // 多线程 HPC 代码一般改用：每个线程独立随机状态 erand48(unsigned short xsubi[3])
  double drand48(); // [0.0, 1.0)
  for ( int i=0; i<m; i++ ) {
    for ( int j=0; j<n; j++ ) {
      // 把 [0,1) 映射到 [-1.0, 1.0)
      A( i,j ) = 2.0 * drand48( ) - 1.0;
    }
  }
}
