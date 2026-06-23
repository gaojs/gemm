/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */
void AddDot( int k, double *x,  double *y, int incy, double *gamma );
void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int i=0; i<m; i+=4 ){ /* Loop over the rows of C */
    for (int j=0; j<n; j+=4 ){ /* Loop over the columns of C, unrolled by 4 */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	   one routine (four inner products) */
      AddDot4x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C */ 

  /* First row */
  AddDot( k, &A( 0, 0 ), &B( 0, 0 ), ldb, &C( 0, 0 ) );
  AddDot( k, &A( 0, 0 ), &B( 0, 1 ), ldb, &C( 0, 1 ) );
  AddDot( k, &A( 0, 0 ), &B( 0, 2 ), ldb, &C( 0, 2 ) );
  AddDot( k, &A( 0, 0 ), &B( 0, 3 ), ldb, &C( 0, 3 ) );

  /* Second row */
  AddDot( k, &A( 1, 0 ), &B( 0, 0 ), ldb, &C( 1, 0 ) );
  AddDot( k, &A( 1, 0 ), &B( 0, 1 ), ldb, &C( 1, 1 ) );
  AddDot( k, &A( 1, 0 ), &B( 0, 2 ), ldb, &C( 1, 2 ) );
  AddDot( k, &A( 1, 0 ), &B( 0, 3 ), ldb, &C( 1, 3 ) );

  /* Third row */
  AddDot( k, &A( 2, 0 ), &B( 0, 0 ), ldb, &C( 2, 0 ) );
  AddDot( k, &A( 2, 0 ), &B( 0, 1 ), ldb, &C( 2, 1 ) );
  AddDot( k, &A( 2, 0 ), &B( 0, 2 ), ldb, &C( 2, 2 ) );
  AddDot( k, &A( 2, 0 ), &B( 0, 3 ), ldb, &C( 2, 3 ) );

  /* Four row */
  AddDot( k, &A( 3, 0 ), &B( 0, 0 ), ldb, &C( 3, 0 ) );
  AddDot( k, &A( 3, 0 ), &B( 0, 1 ), ldb, &C( 3, 1 ) );
  AddDot( k, &A( 3, 0 ), &B( 0, 2 ), ldb, &C( 3, 2 ) );
  AddDot( k, &A( 3, 0 ), &B( 0, 3 ), ldb, &C( 3, 3 ) );
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
