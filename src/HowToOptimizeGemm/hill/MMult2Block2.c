/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_K 32
#define BLOCK_SIZE_N 64

/* Routine for computing C = A * B + C */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int i=0; i<m; i += BLOCK_SIZE_M){ /* Loop over the rows of C */
    int i_end = (i + BLOCK_SIZE_M) < m ? (i + BLOCK_SIZE_M) : m;
    for (int p=0; p<k; p += BLOCK_SIZE_K){ /* Update C( i,j ) with the inner
        product of the ith row of A and the jth column of B */
      int p_end = (p + BLOCK_SIZE_K) < k ? (p + BLOCK_SIZE_K) : k;
      for (int j=0; j<n; j += BLOCK_SIZE_N){ /* Loop over the columns of C */
        int j_end = (j + BLOCK_SIZE_N) < n ? (j + BLOCK_SIZE_N) : n;
        for (int ii = i; ii < i_end; ii++) {
          for (int pp = p; pp < p_end; pp++) {
            register double aip = a[ii * lda + pp];
            for (int jj = j; jj < j_end; jj++) {
              C( ii,jj ) += aip * B( pp,jj );
            }
          }
        }
      }
    }
  }
}


  
