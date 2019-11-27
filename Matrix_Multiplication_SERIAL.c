#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 *	Serial Matrix-Matrix Multiplication
 *
 *	This program uses serial matrix multiplication
 *	and prints the results to the console.
 */

int main(int argc, char **argv) {

	// Global Size for N^2 matrix
	int n = 12;

	// Create Matrix and result arrays to hold integer values
	int *matrix;
	int *result;
	matrix = (int*)malloc(n*n*sizeof(int));
	result = (int*)malloc(n*n*sizeof(int));

	// Create matrix of 2's and print the matrix to the console
	int i = 0;
	while(i < n*n) {
		matrix[i] = 2;
		if(i != 0 && i%n == 0){
			fprintf(stdout, "\n");
		}
		fprintf(stdout, "%d ", matrix[i]);
		i++;
	}
	fprintf(stdout, "\n");
	fprintf(stdout, "\n");

	// Multiply the matrix by itself and store in result matrix
	i = 0;
	while (i < n) {
		int j = 0;
    while (j < n) {
			int k = 0;
      while (k < n) {
      	result[i*n+j] = result[i*n+j] + matrix[i*n+k]*matrix[k*n+j];
        k++;
      }
    	j++;
  	}
    i++;
  }

	// Print the result matrix to console
	i = 0;
	while(i < n*n) {
		if(i != 0 && i%n == 0){
			fprintf(stdout, "\n");
		}
		fprintf(stdout, "%d ", result[i]);
		i++;
	}
}
