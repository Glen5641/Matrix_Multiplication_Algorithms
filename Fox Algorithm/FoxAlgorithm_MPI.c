#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

/*
 * 	Matrix-Matrix Multiplication
 *	Fox Algorithm
 *
 *	Because Fox Algorithm will only work when n divisible by sqrt(p),
 *	given p should be 1, 4, 9, 16, 25, 36, 64,
 *	and n is 100, 200, 300, ..., 1000,
 *	when p is the following, do for the following n:
  * p =  1: 100 : 1000
 *	p =  4: 100 : 1000
 *	p =  9: 300, 600, 900
 *	p = 16: 100 : 1000
 *	p = 25: 100 : 1000
 *	p = 36: 300, 600, 900
 *	p = 64: 200 : 1000 (Even)
 */



#define MATRIXSIZE 1000
#define DEBUG 0

//Print a simple viewable array with
//new lines surrounding it for debugging
void printarr(float* arr, int n){

  fprintf(stdout, "\n");          //Print new line and start loop
  for(int row = 0; row < n*n; row++){

    if(row % n == 0 && row != 0){       //If at end of row, increment row
      fprintf(stdout, "\n");            //and print new line
    }

    //Print each character
    fprintf(stdout, "%6.2f ", arr[row]);
  }

  //Print legibility
  fprintf(stdout, "\n\n");
  return;
}

int main(int argc, char **argv) {

  int comm_sz;        // Number of Processes
  int my_rank;        // Rank as opposed to other processes
	int n = MATRIXSIZE; // Matrix size
  int master = 0;     // Master Rank
  int tag = 0;        // Send and Rec Tag
  float a[n*n];       // Initial A Matrix
  float b[n*n];       // Initial B Matrix
  float matrix[n*n];  // Temp Matrix for partitions
  double start_time;  // Start Time
  double finish_time; // Finish Time
  double final_time;  // Total Time Elapsed

  // Read different number of elements if needed
	if (argc > 1) n = strtol(argv[1], NULL, 10);

	// Create 2 (n by n) matrices with random uniform floats -1:1
	srand(time(NULL));
	float num;
	for (int i = 0; i < n*n; i++) {
		num = (float)rand()/RAND_MAX * 2.0 - 1.0;
    a[i] = num;
    num = (float)rand()/RAND_MAX * 2.0 - 1.0;
	  b[i] = num;
	}

  if(DEBUG) {
    printarr(a, n);           // Initial Matrix print
    printarr(b, n);
	}

	MPI_Init(&argc, &argv);                        // Init Processes
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int np = (int)pow(comm_sz, 0.5);  // Process grid size
  int nr = n/np;                    // E x E per partition
  MPI_Request request;              // Request Status
	MPI_Status status;                // Recieve Status

  if(n%np != 0){         // If n is not divisible by root p, end program
    MPI_Finalize();
    fprintf(stderr, "Square Root of Processes does not divide Number of elements.\n");
    return 0;
  }

  // Time Starts Now!!
	start_time = MPI_Wtime();

	if (my_rank == master) {  // Only Master, Serial Time Partition

    fprintf(stdout, "\nSize: %d\nProcesses: %d\n", n, comm_sz);


    /* Partition matrix where each block is sent to correct process.
    ----------------------------------------------------------------
     0  1  |  2  3          P1:  0  1  4  5
    _4__5__|__6__7_   ==>   P2:  2  3  6  7
     8  9  | 10 11    ==>   P3:  8  9 10 11
    12 13  | 14 15          P4: 12 13 14 15
    ----------------------------------------------------------------
    */
    int i = 0;
    int k = 0;
    for(int row_num = 0; row_num < np; row_num++){
      for(int col_num = 0; col_num < np; col_num++){
        k = col_num*nr + row_num*n*nr;
        for(int j = 0; j < nr*nr; j++){
          if(j % nr == 0 && j != 0){
            k = k + (n-nr);
          }

          // Element Extraction
          matrix[i] = a[k];

          k++;
          i++;
        }
      }
    }

    /* Partition matrix where each block is sent to correct process.
    ----------------------------------------------------------------
     0  1  |  2  3          P1:  0  1  4  5
    _4__5__|__6__7_   ==>   P2:  2  3  6  7
     8  9  | 10 11    ==>   P3:  8  9 10 11
    12 13  | 14 15          P4: 12 13 14 15
    ----------------------------------------------------------------
    */
    i = 0;
    k = 0;
    for(int row_num = 0; row_num < np; row_num++){
      for(int col_num = 0; col_num < np; col_num++){
        k = col_num*nr + row_num*n*nr;
        for(int j = 0; j < nr*nr; j++){
          if(j % nr == 0 && j != 0){
            k = k + (n-nr);
          }

          //Element Extraction
          a[i] = b[k];

          k++;
          i++;
        }
      }
    }
  }

  if(DEBUG) {                 // Print the Content of
    printarr(matrix, n);      // partition matrices if debug
    printarr(b, n);
	}

  float rank_a[nr*nr];  // Create a and b matrices for ranks
  float rank_b[nr*nr];
  float local_a[nr*nr];
  float local_b[nr*nr];

  //Scatter the partition matrices across the comm world
	MPI_Scatter(&a, (nr*nr), MPI_FLOAT, &rank_a, (nr*nr), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&b, (nr*nr), MPI_FLOAT, &rank_b, (nr*nr), MPI_FLOAT, 0, MPI_COMM_WORLD);

  float result[nr*nr];
  for(int i = 0; i < nr*nr; i++){   // Create a result matrix for
    result[i] = 0.0;                // each process to store totals
  }

  /* Grab sources by index - 3 x 3 P Grid example
  ------------------------------------------------
    0  1  2         0  4  8
    3  4  5   -->   1  5  6
    6  7  8         2  3  7
  ------------------------------------------------
  */
  int source[np*np];
  for(int i = 0; i < np; ++i){
    source[i] = i*(np+1);
  }
  for(int i = 1; i < np; ++i){
    for(int j = 0; j < np; ++j){
      if((source[(i-1)*np+j] + 1) >= np*(j+1)){
        source[i*np+j]=np*(j);
      } else {
        source[i*np+j]=source[(i-1)*np+j] + 1;
      }
    }
  }

  // Broadcast the previously collected
  // sources to their close row counterparts
  for(int i = 0; i < np; i++){
    for(int j = 0; j < np; j++){

      for(int k = 0; k < nr*nr; k++){      // For later in program
        local_b[k] = rank_b[k];            // just a swap
      }

      int low = (source[i*np+j]/np);
      int high = (source[i*np+j]/np);   // Secure high and low for the rows
      low = low * np;
      high = (high + 1) * np;

      if(my_rank == source[i*np+j]){     //Pull the source collected earlier
        for(int k = low; k < high; k++){
          if(my_rank != k){

            //Send the element matrix over to the other guys
            if(DEBUG) fprintf(stdout, "Rank: %d Sending A to %d\n", my_rank, k);
            MPI_Send(&rank_a, nr*nr, MPI_FLOAT, k, tag, MPI_COMM_WORLD);

          } else {      // Secure the local matrix for the sender also
            for(int k = 0; k < nr*nr; k++){
              local_a[k] = rank_a[k];
            }
          }
        }
      } else { // If rank is not the source, receive the value
        if(my_rank >= low && my_rank < high){
          if(DEBUG) fprintf(stdout, "Rank: %d Receiving A from %d\n", my_rank, source[i*np+j]);
          MPI_Recv(&local_a, nr*nr, MPI_FLOAT, source[i*np+j], tag, MPI_COMM_WORLD, &status);
        }
      }

      //Do Regular multiplication to the partitioned matrices
      for (int x = 0; x < nr; x++) {
        for (int y = 0; y < nr; y++) {
          for (int z = 0; z < nr ; z++) {
            result[x*nr+y] = result[x*nr+y] + rank_a[x*nr+z]*local_b[z*nr+y];
          }
        }
      }

      //Secure the Destination and the Source for Bubbling B partitions
      int destination = my_rank - np;
      int source = my_rank + np;
      if (destination < 0){
        destination = np*np + destination;
      }
      if (source >= np*np){
        source = source - np*np;
      }

      // Send the B partition to upstairs neighbor.
      if(DEBUG) {
        fprintf(stdout, "Rank: %d Sending B to %d\n", my_rank, destination);
        fprintf(stdout, "Rank: %d Receiving B from %d\n", my_rank, source);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Isend(&rank_b, nr*nr, MPI_FLOAT, destination, 0, MPI_COMM_WORLD, &request);
  		MPI_Irecv(&rank_b, nr*nr, MPI_FLOAT, source, 0, MPI_COMM_WORLD, &request);
  		MPI_Wait(&request, &status);
    }
  }


  MPI_Barrier(MPI_COMM_WORLD);
  finish_time = MPI_Wtime();
  final_time = finish_time - start_time;

  // Pring elapsed time
  if(my_rank == master) {
    fprintf(stdout, "\n"); // Clean Line
    fprintf(stdout, "Total Time Elapsed is %.10f seconds\n", final_time);
  }

  if(DEBUG) {
    int greeting = 1;       //Print Results in an orderly-unorderly fashion
    if(my_rank == master){
      printarr(result, np);
      MPI_Send(&greeting, 1, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD);
    } else if (my_rank == comm_sz - 1) {
      MPI_Recv(&greeting, 1, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, &status);
      printarr(result, np);
    } else {
      MPI_Recv(&greeting, 1, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, &status);
      printarr(result, np);
      MPI_Send(&greeting, 1, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD);
    }
  }

  // Finalize the processes
	MPI_Finalize();

  // End
	return 0;
}
