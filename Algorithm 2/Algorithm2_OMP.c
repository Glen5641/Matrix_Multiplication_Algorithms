#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

/*
 * 	Matrix-Matrix Multiplication
 *	Algorithm 2 with OMP
 *
 *	Because algorithm 2 will only work when n divisible by p,
 *	given p should be 4, 9, 16, 25, 36, 64,
 *	and n is 100, 200, 300, ..., 1000,
 *	when p is the following, do for the following n:
 *	p=4: 100~1000
 *	p=9: 900
 *	p=16: 400, 800
 *	p=25: 100~1000
 *	p=36: 900
 *	p=64: none
 */

 // Global Default Matrix Size and Debug Variable
#define MATRIXSIZE 100
#define DEBUG 0

int main(int argc, char **argv) {

  // Thread Size and Process Rank
  int thread_count = 1, my_rank;

  // If size of matrix is an argument, change matrix size
  // If thread count is an argument, change thread count
  int n = MATRIXSIZE;
  if (argc > 1) {
    n = atoi(argv[1]);
    thread_count = atoi(argv[2]);
    if (DEBUG) printf("User input\n Matrix size: %d \t Threads: %d\n\n", n, thread_count);
	}

  // Declare both matrices and partitions and result matrix
	int matrixA[n][n];
	int matrixB[n][n];
	int partitionA[n*n];
	int partitionB[n*n];
	int partitionC[n*n];
	int matrixC[n][n];

  // Declare start, finish, and final times
	double start_time, finish_time, final_time;

	// Create random elements in matrices (0-9)
	srand(time(NULL));
	int num;
	for (int i = 0; i < n; i++) {
	  for (int j = 0; j < n; j++) {
			num = (rand() % 10);
			matrixA[i][j] = num;
			num = (rand() % 10);
			matrixB[i][j] = num;
		}
	}

	// Exit if n is not divisble by p
	if (n%thread_count != 0) {
		printf("Matrix size %d not divisible by process count %d\n\n", n, thread_count);
		return 0;
	}

  // Start Parallel calculations
	start_time = omp_get_wtime();

	// Change matrixA into one-dimension array by row
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			partitionA[(row*n)+col] = matrixA[row][col];
		}
	}

	// Change matrixB into one-dimension array by col
	for (int col = 0; col < n; col++) {
		for (int row = 0; row < n; row++) {
			partitionB[(col*n)+row] = matrixB[row][col];
		}
	}

	if (DEBUG) printf("Begin calculations\n");

  // Set Num Threads from OMP
	omp_set_num_threads(thread_count);

  // Each process will contain columns of C
	int localCalcA[n*n/thread_count];
	int localCalcB[n*n/thread_count];
	int localCalcC[n*n/thread_count];

  // Zero out partitionC and localCalcC for easier debugging
	if (DEBUG) {
		for (int i = 0; i < n*n/thread_count; i++) {
			localCalcC[i] = 0;
		}
		for (int i = 0; i < n*n; i++) {
			partitionC[i] = 0;
		}
	}

  // Declare thread id and sum vars
	int tid;
	int sum = 0;

	// All processes do their respective calculations
# pragma omp parallel private(tid, localCalcA, localCalcB, localCalcC, sum)
	for (int iter = 0; iter < thread_count; iter++) {

    // Form localCalcB that is held the first run
		if (iter == 0) {
			tid = omp_get_thread_num();
			for (int i = 0; i < n*n/thread_count; i++) {
				localCalcB[i] = partitionB[(n*n/thread_count)*tid+i];
			}
			if (DEBUG) {
# pragma omp critical
				{
					printf("Thread %d finished localCalcB\n", tid);
					for (int j = 0; j < n*n/thread_count; j++)
						printf("%2d", localCalcB[j]);
					printf("\n");
				}
			}
		}

		// Form localCalcA which will change each iteration like a ring pass
		for (int i = 0; i < n*n/thread_count; i++) {
			localCalcA[i] = partitionA[(n*n/thread_count)*((tid+iter)%thread_count)+i];
		}
		if (DEBUG) {
# pragma omp critical
			{
				printf("Thread %d finished localCalcA\n", tid);
				for (int j = 0; j < n*n/thread_count; j++)
					printf("%2d", localCalcA[j]);
				printf("\n");
			}
		}

		sum = 0;

    // Traverse local partitions row, multiply and add into C element
		for (int i = 0; i < n/thread_count; i++) {	// how many rows per process
			for (int k = 0; k < n/thread_count; k++) { 	// how many columns per process
				for (int f = 0; f < n; f++) {	// traverse partition
					sum += localCalcA[f+i*n]*localCalcB[f+k*n];
				}
# pragma omp critical
				{
					partitionC[k*n+i+((tid+iter)%thread_count)*n/thread_count+(tid*n*n/thread_count)] = sum;
					if (DEBUG) {
						printf("Thread %d: Sum: %3d partitionC[%2d]: %d\n", tid, sum, k*n+i+((tid+iter)%thread_count)*n/thread_count, partitionC[k*n+i+((tid+iter)%thread_count)*n/thread_count+(tid*n*n/thread_count)]);
					}
					sum = 0;
				}
			}
		}
		if (DEBUG) {
# pragma omp critical
			{
				printf("Thread %d finished iter %d\n", tid, iter);
				for (int j = 0; j < n*n/thread_count; j++)
					printf("%4d", partitionC[j]);
				printf("\n");
			}
		}
	}

	if (DEBUG) printf("Completed calculations\n");

  // Gather Partition C to Matrix C
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matrixC[j][i] = partitionC[(i*n)+j];
		}
	}

  // Set finish time
	finish_time = omp_get_wtime();
	 
	if (DEBUG) printf("Finished matrix.\n");

	// Validation
	if (n <= 8) {		// only print out for validation if matrix size is reasonable (n<=8)
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("%2d", matrixA[i][j]);
			}
			printf("   ");
			for (int j = 0; j < n; j++) {
				printf("%2d", matrixB[i][j]);
			}
			printf("   ");
			for (int j = 0; j < n; j++) {
				printf("%4d", matrixC[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

  // Print Time
	final_time = finish_time - start_time;
	printf("Threads: %2d \t Matrix Size: %4d \t Final Time: %f\n\n", thread_count, n, final_time);

	return 0;
 }
