#  Distributed MatMul for high-common dimension

The current distributed block MM algorithms such as Cannon's (https://iq.opengenus.org/cannon-algorithm-distributed-matrix-multiplication/) and Summa (http://www.netlib.org/lapack/lawnspdf/lawn96.pdf) are not optimized for processor memory constraints or data movement for a certain class of matrices where common "N" dimension is very large.

I developed this algorithm (called mosaic) which is targeted for both square and rectangular matrices, and blocks are square or rectangular. 
There are two key ideas in the algorithm:

1. Based on the memory available in the processor and number of processors, matrix is partitioned into rigth-size rectangular blocks for LHS and RHS. 
   Matrix blocks are replicated as much as possilbe to avoid unnecessary exchanges. If there is not enough memory, the RHS matrix is split up for exchanges.
   Note: exchanges are the outermost loop and costly. Note that there are two distinct blocks by dimensions due to the rectangular sizes (unlike Cannon's). 
   
2. Only one of the two blocks moves around during computation. The other one consistently stays in the same processor. This is in contrast to Cannon's and Summa. This decreases size of data movement.
   However, it means that reduction in  $\sum$ (W[i,j]*X[j,k]) for j in (1:N) cannot be done in the same processor. The algorithm reduces by recursive halving it across all tiles in the common dimension.
   
   

## Complexity Analysis of Algorithm

The  complexity of this algorithm is virtually the same as any other block-MM except the reduction part. Consider, LHS is W (dimensions MxN). RHS is X (dimensions NxP). 
Output is Y (dimensions MxP).

The operation is: Y = W dot X  

Consider a number of processors with combined GFLOPs of "F" when running in parallel.
Consider bandwidth between any two processors is fixed at B (GB/sec).

For the impractical case of block size 1x1 and with infinite memory, 

Time is approximated as O(M*P*(2*N-1)/F + 1*1*logN/B) nSec (or mSec if M/P/N are in millions).
So, there are no "exchanges" happening as memory is limitless. The only movement is for summation of all results in the N-dimension.

Let the LHS block be m x n, and RHS block be n x p.

The total time = O(M*P*2*(N-1)/F  + m*p*log(N/n)/ B + xc*n*p/B)
Here "xc" refers to the exchanges required in case the memory is not enough to fit all of RHS matrix X. We split the RHS into groups of "Xc" columns. 
Each Xc processors working on contiguous columns only have separate copies of Xc. After every reduction, a circular exchange between the Xc processsors produces the next result. 

## Algorithm:

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

## Optimizer:

The optimizer that determines the partitioning, the block sizing (m, n, p), and the number of exchanges (Xc) is not perfect here. 
For instance, for 2048x2048 square MM, it only engages 512 processors out of 800 possible processors. 

More work has to be done on optimizer. If the residual group problem is solved, the optimizer naturally improves. But need a better
solver than the iterative method in here (perhaps, look into scipy).


