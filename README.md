# Matrix - Matrix Multiplication

## Description
### Algorithm 2
Perform Matrix Multiplication by sending the corresponding row and  
its column to each process. The process does the calculation and  
sends the answer back to the master process.  

### Fox Algorithm
Perform Matrix Multiplication by Taking the processes as a matrix like  
algorithm 2, but instead, multiplies each element and holds the value  
in the process while shifting matrix a and bubbling up matrix b.  

## Files
| File                         | Language | Library | Description                                              |
|------------------------------|:--------:|---------|----------------------------------------------------------|
| Algorithm2_MPI               | C        | MPI     | Processes information on CPU processes (Unshared Memory) |
| Algorithm2_OMP               | C        | OMP     | Processes information on CPU threads   (shared Memory)   |
| FoxAlgorithm_MPI             | C        | MPI     | Processes information on CPU processes (Unshared Memory) |
| FoxAlgorithm_OMP             | C        | OMP     | Processes information on CPU threads   (shared Memory)   |
| Matrix_Multiplication_SERIAL | C        | SERIAL  | Processes information on 1 CPU Process                   |
