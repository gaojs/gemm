/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

void dot4x4(double *a, int lda, double *b, int ldb,double *c, int ldc)
{
  register double aip0 = A( 0,0 );
  register double aip1 = A( 1,0 );
  register double aip2 = A( 2,0 );
  register double aip3 = A( 3,0 );

  // 计算C的4x4子块
  C(0, 0) += aip0 * B(0, 0);
  C(0, 1) += aip0 * B(0, 1);
  C(0, 2) += aip0 * B(0, 2);
  C(0, 3) += aip0 * B(0, 3);
  C(1, 0) += aip1 * B(0, 0);
  C(1, 1) += aip1 * B(0, 1);
  C(1, 2) += aip1 * B(0, 2);
  C(1, 3) += aip1 * B(0, 3);
  C(2, 0) += aip2 * B(0, 0);
  C(2, 1) += aip2 * B(0, 1);
  C(2, 2) += aip2 * B(0, 2);
  C(2, 3) += aip2 * B(0, 3);
  C(3, 0) += aip3 * B(0, 0);
  C(3, 1) += aip3 * B(0, 1);
  C(3, 2) += aip3 * B(0, 2);
  C(3, 3) += aip3 * B(0, 3);
}

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
  double *b, int ldb,double *c, int ldc )
{
  for (int p=0; p<k; p++ ){
    for (int i=0; i<m; i+=4 ){
      for (int j=0; j<n; j+=4 ){
        dot4x4(&A(i,p), lda, &B(p,j), ldb, &C(i,j), ldc);
      }
    }
  }
}
 
