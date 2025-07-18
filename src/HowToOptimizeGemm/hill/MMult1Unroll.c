/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int i=0; i<m; i++ ){
    for (int p=0; p<k; p++ ){
      register double aip = A( i,p );
      for (int j=0; j<n; j+=4 ){
        C( i,j+0 ) +=  aip * B( p,j+0 );
        C( i,j+1 ) +=  aip * B( p,j+1 );
        C( i,j+2 ) +=  aip * B( p,j+2 );
        C( i,j+3 ) +=  aip * B( p,j+3 );
      }
    }
  }
}


  
