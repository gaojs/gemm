#include <stdlib.h>

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]
// 定义矩阵访问宏（行主序）
#define AT(i,j) at[(i)*ldat + (j)]

// 矩阵转置函数
double* transpose(int m, int k, double* a, int lda, int *ldat_ptr) {
    double* at = (double*)malloc(m * k * sizeof(double));
    if (!at) return 0;
    int ldat = m;
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            AT(p, i) = A(i, p);
        }
    }
    *ldat_ptr = ldat;
    return at;
}

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int ldat = lda;
  double *at = transpose(m, k, a, lda, &ldat);
  for (int i=0; i<m; i++ ){ /* Loop over the rows of C */
    for (int p=0; p<k; p++ ){ /* Update C( i,j ) with the inner
        product of the ith row of A and the jth column of B */
      for (int j=0; j<n; j++ ){ /* Loop over the columns of C */
        C( i,j ) = C( i,j ) +  AT( p,i ) * B( p,j );
      }
    }
  }
  free(at);
}


  
