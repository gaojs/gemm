/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */
void AddDot( int k, double *x,  double *y, int incy, double *gamma );
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int i=0; i<m; i+=1 ){ /* Loop over the rows of C */
    for (int j=0; j<n; j+=1 ){ /* Loop over the columns of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
        and the jth column of B */
      AddDot( k, &A( i,0 ), &B( 0,j ), ldb, &C( i,j ) );
    }
  }
}


/* Create macro to let Y(i) equal the ith element of y */
#define Y(i) y[ (i)*incy ]
void AddDot( int k, double *x,  double *y, int incy, double *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.
     Here x starts at location x and has (implicit) stride of 1 and 
        y starts at location y with increment (stride) incy.
  */
  for (int p=0; p<k; p++ ){
    *gamma += x[p] * Y(p);     
  }
}
