#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"

void REF_MMult( int m, int n, int k, double *a, int lda, 
                double *b, int ldb, double *c, int ldc );
void MY_MMult( int m, int n, int k, double *a, int lda, 
                double *b, int ldb, double *c, int ldc );
void copy_matrix( int m, int n, double *a, int lda, double *b, int ldb );
void random_matrix( int m, int n, double *a, int lda );
double compare_matrices( int m, int n, double *a, int lda, double *b, int ldb );

double dclock();

int main()
{
  int p, m, n, k, lda, ldb, ldc, rep;
  double dtime, dtime_best, gflops, diff;
  double *a, *b, *c, *c_ref, *c_old;    
  
  printf( "MY_MMult = [\n" );    
  for ( p=PFIRST; p<=PLAST; p+=PINC ){
    m = ( M == -1 ? p : M );
    n = ( N == -1 ? p : N );
    k = ( K == -1 ? p : K );

    lda = ( LDA == -1 ? m : LDA );
    ldb = ( LDB == -1 ? k : LDB );
    ldc = ( LDC == -1 ? m : LDC );

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( double * ) malloc( lda * (k+1) * sizeof( double ) );  
    b = ( double * ) malloc( ldb * n * sizeof( double ) );
    c = ( double * ) malloc( ldc * n * sizeof( double ) );
    c_old = ( double * ) malloc( ldc * n * sizeof( double ) );
    c_ref = ( double * ) malloc( ldc * n * sizeof( double ) );

    /* Generate random matrices A, B, c_old */
    random_matrix( m, k, a, lda );
    random_matrix( k, n, b, ldb );
    random_matrix( m, n, c_old, ldc );

    copy_matrix( m, n, c_old, ldc, c_ref, ldc );

    /* Run the reference implementation so the answers can be compared */
    REF_MMult( m, n, k, a, lda, b, ldb, c_ref, ldc );

    /* Time the "optimized" implementation */
    for ( rep=0; rep<NREPEATS; rep++ ){
      copy_matrix( m, n, c_old, ldc, c, ldc );

      /* Time your implementation */
      dtime = dclock();
      MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );      
      dtime = dclock() - dtime;

      if ( rep==0 ) {
        dtime_best = dtime;
      } else {
        dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
      }
    }
    gflops = 2.0 * m * n * k * 1.0e-09;
    diff = compare_matrices( m, n, c, ldc, c_ref, ldc );
    printf( "%d %le %le \n", p, gflops / dtime_best, diff );
    fflush( stdout );

    free( a );
    free( b );
    free( c );
    free( c_old );
    free( c_ref );
  }

  printf( "];\n" );
  exit( 0 );
}