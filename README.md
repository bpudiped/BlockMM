
# new distributed MatMul algorithm

Initial check-in.

###  Overview

The current distributed block MM algorithms such as Cannon's (https://iq.opengenus.org/cannon-algorithm-distributed-matrix-multiplication/) and Summa (http://www.netlib.org/lapack/lawnspdf/lawn96.pdf) are not optimized for processor memory constraints or data movement for a certain class of matrices where common "N" dimension is very large.

This algorithm (called mosaic) was developed for both square and rectangular matrices, and the blocks too are square or rectangular (leading to a lot of possible scenarios for the optimizer).

There are two key ideas in the algorithm:

1. Based on the memory available in the processor and number of processors, matrix is partitioned into rigth-size rectangular blocks for LHS and RHS. 
   Matrix blocks are replicated as much as possilbe to avoid unnecessary exchanges. If there is not enough memory, the RHS matrix is split up for exchanges.
   Note: exchanges are the outermost loop and costly. Note that there are two distinct blocks by dimensions due to the rectangular sizes (unlike Cannon's). 
   
2. Only one of the two blocks moves around during computation. The other one consistently stays in the same processor. This is in contrast to Cannon's and Summa. This decreases size of data movement.
   However, it means that reduction in  $\sum$ (W[i,j]*X[j,k]) for j in (1:N) cannot be done in the same processor. The algorithm reduces by recursive halving it across all tiles in the common dimension.
   
   

### Complexity Analysis of Algorithm

The  complexity of this algorithm is virtually the same as any other block-MM except the reduction part. Consider, LHS is W (dimensions M x N). RHS is X (dimensions N x P). 
Output is Y (dimensions MxP).

The operation is: Y = W dot X  

Consider a number of processors with combined GFLOPs of "F" when running in parallel.
Consider bandwidth between any two processors is fixed at B (GB/sec).

For the impractical case of block size 1x1 and with infinite memory (assuming FP32).

Time is approximated as O(M x P x (2N-1)/F + 4 x 1 x 1 x logN/B) nSec (or mSec if M/P/N are in millions).
So, there are no "exchanges" happening as memory is limitless. The only movement is for summation of all results in the N-dimension.

Let the LHS block be m x n, and RHS block be n x p.

The total time = O(M x P x 2(N-1)/F  + 4 x m x p x log(N/n)/ B + 4 x Xc x n x P/B)
Here "Xc" refers to the exchanges required in case the memory is not enough to fit all of RHS matrix X. We split the RHS into groups of "Xc" columns. 
Each Xc processors working on contiguous columns only have separate copies of Xc. After every reduction, a circular exchange between the Xc processsors produces the next result. 

### Algorithm

The main ideas are described in the two figures. The first figure shows the arrangement of the data. 
So each processor has infinite memory and gets an mxn block (from matrix W) and an nxp block (from matrix X). 
There is infinite memory, so blocks are big enough for no exchanges and small enough to occupy all processors.
All processors compute their results in the first step. The second step is the reduction across the N-dimension.

![Reduction by Recursive Halving](https://github.com/bpudiped/MosaicMM/blob/master/mosiacMM1.png)

In the second figure, there isn't enough memory to split the matrix to produce all the results in one compute iteration.
The second matrix is split up into "exchange groups." A processor group holds contiguous columns of an exchange group. 
After every compute/reduction iteration, a circular exchange ensues between the processors in the same exchange group.
After "Xc" number of exchanges, the MM operation is done. The value of Xc is chosen by the optimizer as part of maximizing
compute and fitting memory constraints solver. 

![Exchange of Matrix X-blocks](https://github.com/bpudiped/MosaicMM/blob/master/mosiacMM.png)

The current algorithm does not work on some irregular sized matrices covered by assertions. That issue will be fixed without
any performance impact (the "residual group" problem).

### Optimizer

The optimizer that determines the partitioning, the block sizing (m, n, p), and the number of exchanges (Xc) is not perfect here. 
For instance, for 2048x2048 square MM, it only engages 512 processors out of 800 possible processors. 

More work has to be done on optimizer. If the residual group problem is solved, the optimizer naturally improves. But need a better
solver than the iterative method in here (perhaps, look into scipy).

The ideal optimizer should be a multi-variable constraint solver for m, n, p, Xch such that number of processors and processor memory
are close to maximum values without going over.

### Simulating Processors

The algorithm is run on "simulated" processors using a simple abstraction where a processor can multiply a block with a certain efficiency (80% by default).
The loss in efficiency is due to memory accesses needed for reloading or setting up the SIMD or systolic array. 

The bandwidth, frequency, and fmacs (width of the simd or systolic array) can be specified. An fmac performs a multiply and an add. So, in a reduction, 
the multiply is unused.

This approximation is fairly acceptable in a cluster with uniform P2P bandwidth. For multi-bandwidth

The code finds the partitioning, initializes the processors, and runs the algorithm that includes computation, reduction and exchanges.
A cycle count is returned for each task (computation, reduction, exchange). With the frequency, the effective TFLOPs is measured.

### Comparison with Other Algorithms

TBD. The same appartus will be used to run Cannon's and probably other algorithms like Summa.

### Sample output

![2688-dimension square matrix multiplication](https://github.com/bpudiped/MosaicMM/blob/master/mosaicLog.PNG)





