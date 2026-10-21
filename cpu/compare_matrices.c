#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define abs( x ) ( (x) < 0.0 ? -(x) : (x) )

double compare_matrices( int m, int n, double *a, int lda, double *b, int ldb )
{
  double max_diff = 0.0;
  for (int i=0; i<m; i++ ){
    for (int j=0; j<n; j++ ){
      double diff = abs( A( i,j ) - B( i,j ) );
      max_diff = ( diff > max_diff ? diff : max_diff );
    }
  }
  return max_diff;
}

