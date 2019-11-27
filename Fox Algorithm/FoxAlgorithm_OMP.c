#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

/*
 * Project 2 - Fox Algorithm With Open MP
 * Authors: Group 2
 *
 * This Project uses OpenMP and thread
 * topology to multiply to N*N matrices.
 */

// Do not Debug With Large Matrices
#define DEBUG 0
#define MAXSIZE 1000
#define MINSIZE 2
#define MINTHREADS 1
#define MAXTHREADS 64

/*
 * Helper Function to print a Matrix Variable as
 * --         --
 * | 0 1 2 3 4 |
 * | 0 1 2 3 4 |
 * | 0 1 2 3 4 |
 * --         --
 */
void print_matrix(int m, int n, double (*matrix)[n]){

  // Clean Print
  fprintf( stdout, "\n" );

  // Iterate through all elements and print with padding
  for( int row = 0; row < m; row++ ) {
    for( int col = 0; col < n; col++ ) {
      fprintf( stdout, "%6.2f ", matrix[row][col] );
    }

    // Finished Row, Go new line
    fprintf( stdout, "\n" );
  }

  //Clean Print
  fprintf( stdout, "\n" );
  return;
}

/*
 * Function to Mutate Matrix to Array that
 * will be mutated again to a thread matrix
 *
 * | 0 0 1 1 |  =>  Thread 0 | 0 0 0 0 |
 * | 0 0 1 1 |  =>  Thread 1 | 1 1 1 1 |
 * | 2 2 3 3 |  =>  Thread 2 | 2 2 2 2 |
 * | 2 2 3 3 |  =>  Thread 3 | 3 3 3 3 |
 */
double* mutate_to_thread_array(int n, int t, double (*matrix)[n]){

  // Allocate Array Pointer and Get Local Variables
  int n_per_thread = (n*n)/t;
  int nr_per_thread = (int)pow(((n*n) / t), 0.5);
  double mat[t][n_per_thread];
  double *arr;
  arr = (double*)malloc(n*n*sizeof(double));

  // Iterate Through Matrix and Seperate Thread matrix
  for(int iter = 0; iter < nr_per_thread; iter++){
    int r = -1;
    int c = iter * nr_per_thread;
    for(int i = iter; i < n; i += nr_per_thread){
      for(int j = 0; j < n; j++){
        if(j % nr_per_thread == 0) {
          r++;
          c = iter * nr_per_thread;
        }
        mat[r][c] = matrix[i][j];
        c++;
      }
    }
  }

  // Stack the Columns side by side
  for(int i = 0; i < t; i++){
    for(int j = 0; j < n_per_thread; j++){
      arr[(i*n_per_thread)+j] = mat[i][j];
    }
  }

  return arr;
}

int main( int argc, char **argv ) {

  // Check if Args are There
  if ( argc < 3 || argc > 3 ) {
    fprintf(stderr, "Invalid Arguments\nCorrect Form: ./fox [# Threads] [n elements]\n");
    return (-1);
  }

  // Get Threads and Size from Args
  int n_threads = strtol( argv[1], NULL, 10 );
  int n         = strtol( argv[2], NULL, 10 );

  // # Threads must be within MIN and MAX
  if( n_threads < MINTHREADS || n_threads > MAXTHREADS ) {
    fprintf( stderr, "Invalid Thread Count %d\n", n_threads );
    return (-1);
  }

  // N must be within MIN and MAX
  if( n < MINSIZE || n > MAXSIZE ) {
    fprintf( stderr, "Invalid N %d\n Should be b.\n", n );
    return (-1);
  }

  // Print Identifying Run
  fprintf( stdout, "Thread Count: %d\n", n_threads );
  fprintf( stdout, "N Elements:   %d\n", n );

  // Entering Serial
  if( DEBUG ) fprintf( stdout, "Entering Serial\n" );

  // The Grid of Threads has Row size of square root of threads
  int thread_row = ( int ) pow( n_threads , 0.5 );

  // Check if Fox Algorithm is applicable
  if( n % thread_row != 0 ){
    fprintf( stderr, "Square Root of Threads does not Divide Number of Elements.\n" );
    return (0);
  }

  // Create original A and B matrices and populate them
  double a_original[n][n];
  double b_original[n][n];
  srand( time( NULL ) );
  for( int i = 0; i < n; i++ ) {
    for( int j = 0; j < n; j++ ) {
      double num;
      num = ( double )rand()/RAND_MAX * 2.0 - 1.0;
      a_original[i][j] = num;
      num = ( double )rand()/RAND_MAX * 2.0 - 1.0;
      b_original[i][j] = num;
    }
  }

  // Print original Matrices
  if(DEBUG){
    print_matrix(n, n, a_original);
    print_matrix(n, n, b_original);
  }

  // Mutate the Matrices into Arrays
  double* a_original_arr = mutate_to_thread_array(n, n_threads, a_original);
  double* b_original_arr = mutate_to_thread_array(n, n_threads, b_original);

  // Gather Variables for calculations
  int npt = n*n/n_threads;
  int nptr = (int)pow(npt, 0.5);

  // Create A and B Matrices where Each Row is to a thread
  double a[n_threads][npt];
  double b[n_threads][npt];

  // Create Result 3d Matrix Where Each Thread now has its own Matrix
  // Memory Issue Where Data is getting overwritten.
  // Solved with Extra Allocated Space
  double result[n_threads*2][nptr][nptr];

  // Initiallize Result to Zero
  for(int c = 0; c < n_threads; c++){
    for(int d = 0; d < nptr; d++){
      for(int e = 0; e < nptr; e++){
        result[c][d][e] = 0.0;
      }
    }
  }

  // Transfer The Thread Mutated Array into the A and B matrices
  int y = -1, z = 0;
  for( int i = 0; i < n*n; i++ ) {
    if(i % (n*n/n_threads) == 0){
      y++;
      z = 0;
    }
    a[y][z] = a_original_arr[i];
    b[y][z] = b_original_arr[i];
    z++;
  }

  // Print Thread Matrix
  if(DEBUG) {
    print_matrix(n_threads, npt, a);
    print_matrix(n_threads, npt, b);
  }

  // Create Needed Variables for Fox Calc
  double a_mat[nptr][nptr];
  double b_mat[nptr][nptr];
  int next = 0;
  int a_mul = 0;
  double temp[n_threads][npt];

  if(DEBUG) {
    printf("Entering Parallel\n");
  }

  // Start the Parallel Time
  double start = omp_get_wtime();

  // Iterate Through Calculations Thread Row Times to ensure Loop back
  for(int iter = 0; iter < thread_row; iter++){
    next = 0;

    // Iterated through the Threads Passing A Values to Correct Thread Rows
    for(int i = iter; i < n_threads; i += thread_row){
      a_mul = i+next;

      // Parallellize The Finding the Correct A Values and Multiplying
      // that A val with B val and Summing the result to Result Matrix
      #pragma omp parallel for num_threads(n_threads) private(a_mat, b_mat)
      for(int j = (thread_row*next); j < (thread_row*next)+thread_row; j++){

        // Find A Matrix to Pass to Thread Row
        if(i != 0){
          if((a_mul / ((next+1)*thread_row)) == 1){
            int num = a_mul % ((next+1)*thread_row);
            a_mul = (thread_row*next)+num;
          }
        }

        // Store the Correct A and Corresponding B to Local Thread Matrices
        for(int index = 0; index < npt; index++){
          a_mat[index/nptr][index%nptr] = a[a_mul][index];
          b_mat[index/nptr][index%nptr] = b[j][index];
        }

        // Multiply the Smaller Matrices using Matrix Multiplication
        // And Store Local Result Matrix
        for (int c = 0; c < nptr; c++) {
          for (int d = 0; d < nptr; d++) {
            double sum = 0.0;
            for (int e = 0; e < nptr; e++) {
              sum = sum + a_mat[c][e]*b_mat[e][d];
            }
            result[j][c][d] = result[j][c][d] + sum;
          }
        }
      }
      next++;
    }

    // Transfer The B Vals to Temp Matrix
    #pragma omp parallel for num_threads(n_threads) shared(temp, b)
    for(int j = 0; j < n_threads; j++){
      for(int k = 0; k < npt; k++){
        temp[j][k] = b[j][k];
      }
    }

    // Transfer back to B While Bubbling Up values and Sending top to bottom
    #pragma omp parallel for num_threads(n_threads) shared(temp, b)
    for(int j = 0; j < n_threads; j++){
      for(int k = 0; k < npt; k++){
        if(j < thread_row){
          b[(thread_row*(thread_row-1)+j)][k] = temp[j][k];
        }
        b[j-thread_row][k] = temp[j][k];
      }
    }
  }

  /* Grab the Final Time and Print to Standard Out */
  double final_time = omp_get_wtime() - start;
  fprintf(stdout, "RunTime: %2.10f seconds\n\n", final_time);

  // Print Result Matrices as
  // --                                                                  --
  // |  (Result 1)                            ...  (Result Thread_Row)    |
  // |  (Result Thread_Row+1)                 ...  (Result 2*Thread_Row)  |
  // |  ................................................................  |
  // |  (Result Thread_Row*(Thread_Row - 1))  ...  (Result Thread_Row^2)  |
  // --                                                                  --
  if(DEBUG){
    for(int i = 0; i < n_threads; i++){
      print_matrix(nptr, nptr, result[i]);
    }
  }
}
