/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int p=0; p<k; p++ ){
    for (int i=0; i<m; i+=4 ){
      register double aip0 = A( i+0,p );
      register double aip1 = A( i+1,p );
      register double aip2 = A( i+2,p );
      register double aip3 = A( i+3,p );
      for (int j=0; j<n; j+=4 ){
        C( i+0,j+0 ) +=  aip0 * B( p,j+0 );
        C( i+0,j+1 ) +=  aip0 * B( p,j+1 );
        C( i+0,j+2 ) +=  aip0 * B( p,j+2 );
        C( i+0,j+3 ) +=  aip0 * B( p,j+3 );
        C( i+1,j+0 ) +=  aip1 * B( p,j+0 );
        C( i+1,j+1 ) +=  aip1 * B( p,j+1 );
        C( i+1,j+2 ) +=  aip1 * B( p,j+2 );
        C( i+1,j+3 ) +=  aip1 * B( p,j+3 );
        C( i+2,j+0 ) +=  aip2 * B( p,j+0 );
        C( i+2,j+1 ) +=  aip2 * B( p,j+1 );
        C( i+2,j+2 ) +=  aip2 * B( p,j+2 );
        C( i+2,j+3 ) +=  aip2 * B( p,j+3 );
        C( i+3,j+0 ) +=  aip3 * B( p,j+0 );
        C( i+3,j+1 ) +=  aip3 * B( p,j+1 );
        C( i+3,j+2 ) +=  aip3 * B( p,j+2 );
        C( i+3,j+3 ) +=  aip3 * B( p,j+3 );
      }
    }
  }
}
 
