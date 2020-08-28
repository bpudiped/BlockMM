
# new distributed MatMul algorithm

Initial check-in.

###  Overview

This algorithm (called mosaic) is targeted for a special class of square and rectangular matrices more often seen in large NLP problems. The blocks too are square or rectangular (leading to a lot of possible scenarios for the optimizer). It is simulated and verified by comparing its results with numpy.matmul. The performance is estimated from cycle counts coming from the simulated processors.

Some block MM algorithms such as [Cannon](https://iq.opengenus.org/cannon-algorithm-distributed-matrix-multiplication/) and [Summa](http://www.netlib.org/lapack/lawnspdf/lawn96.pdf) are not optimized for processor memory constraints or data movement for matrices of the type where common "N" dimension is very large.

While the genrealized form of Cannon's original algorithm works with rectangular matrices, Cannon's overhead cost involves shifting two blocks after every synchronized iteration. This pays a heavy penalty in huge matrices with large / large characteristics i.e. large number of processors, and large block sizes (as the matrices are gigantic).

I hadn't read about Summa when I designed Mosaic, but Mosaic seems to have many things in common with Summa (a high-level overview of Summa is [here](http://cseweb.ucsd.edu/classes/fa12/cse260-b/Lectures/Lec13.pdf)) 

Yet, there are diffferences too. One of the two matrices (either LHS or RHS) never moves in Mosaic. This property is useful when choosing block sizes (mxn and nxp) so that the stationary matrix may have a large block size. 

There are also two other key ideas in the algorithm:

1. Based on the memory available in the processor and number of processors, matrix is partitioned into right-size rectangular blocks for LHS and RHS. 
   Matrix blocks are replicated as much as possible to avoid unnecessary exchanges. If there is not enough memory, the RHS matrix is split up for exchanges.
   Note: exchanges are the outermost loop (not shown in the previous loops) and costly. Note that there are two distinct blocks by dimensions due to the rectangular sizes (unlike Cannon's). 
   
2. Only one of the two blocks moves around during computation. The other one consistently stays in the same processor. This is in contrast to Cannon's and Summa. This decreases size of data movement.

   However, it means that reduction in cannot be done in the same processor. 
   The algorithm, therefore, uses a simple binary reduction for the outermost loop (i.e. j in 0 to N-1)
   
With exchanges, the loops at a high-level are:

// Initialize Y to all zeros
// Xc is number of exchanges determined by optimizer to fit memory contraints

   Px = P/Xc
   
   for (e=0; e < Xc; ++ e):
   
      for (j=0; j<N; ++j):
      
         for (i=0; i<M; ++i): 
         
            for (k=e*Px; k< (e+1)*Px; ++k): 
            
               Y[i, k] += W[i,j] x X[j, k]
               
      exchange(e)  // routine that exchanges blocks of X (within its group)
            
NOTE: The simple implementation here is on a simulator and does not cover some very odd irregular sized matrices. Also, lacking an apples-to-apples comparision with Cannon's and Summa's (i.e running those algorithms on the same simulated processors).

### Complexity

The  complexity of this algorithm is virtually the same as any other block-MM except the reduction part. Consider, LHS is W (dimensions M x N). RHS is X (dimensions N x P). 
Output is Y (dimensions MxP).

The operation is: Y = W dot X  

Consider a number of processors with combined GFLOPs of "F" when running in parallel.
Consider bandwidth between any two processors is fixed at B (GB/sec).

For the impractical case of block size 1x1 and with infinite memory (assuming FP32).

Time is approximated as O(M x P x (2N-1)/F + 4 x 1 x 1 x logN/B) nSec (or mSec or secs if M x P x N is in millions or billions).
So, there are no "exchanges" happening as memory is limitless. The only movement is for summation of all results in the N-dimension.

Let the LHS block be m x n, and RHS block be n x p.

The total time = O(M x P x 2(N-1)/F  + 4 x m x p x log(N/n)/ B + 4 x Xc x n x P/B)
Here "Xc" refers to the exchanges required in case the memory is not enough to fit all of RHS matrix X. We split the RHS into groups of "Xc" columns. 
The RHS matrix is divided into groups of Xc columns each. After every reduction, a circular exchange between the Xc columns produces the next result. 

### Algorithm details

The main ideas are described in the two figures. The first figure shows the arrangement of the data. 
So each processor has infinite memory and gets an mxn block (from matrix W) and an nxp block (from matrix X). 
There is infinite memory, so blocks are big enough for no exchanges and small enough to occupy all processors.
All processors compute their results in the first step. The second step is the reduction across the N-dimension.

![Reduction in log-N steps](https://github.com/bpudiped/MosaicMM/blob/master/mosiacMM2.png)

In the second figure, there isn't enough memory to split the matrix to produce all the results in one compute iteration.
The second matrix is split up into "exchange groups." A processor group holds contiguous columns of an exchange group. 
After every compute/reduction iteration, a circular exchange ensues between the processors in the same exchange group.
After "Xc" number of exchanges, the MM operation is done. The value of Xc is chosen by the optimizer as part of maximizing
compute and fitting memory constraints solver. 

![Exchange of Matrix X-blocks](https://github.com/bpudiped/MosaicMM/blob/master/mosiacMM.png)

### Optimizer

The optimizer that determines the partitioning, the block sizing (m, n, p), and the number of exchanges (Xc) is not perfect here. 
The ideal optimizer should be a multi-variable constraint solver for m, n, p, Xch such that number of processors and processor memory
are close to maximum values without going over limits (maybe using a scipy minimization library).

The current implementation uses an iterative strategy by first getting as close to max processors as it can. 
Then, if we are still over "per-processor memory limit", it scales back memory by increasing exchanges. 

### Simulating Processors

The algorithm is run on "simulated" processors using a simple abstraction where a processor can multiply a block with a certain efficiency (80% by default).
The loss in efficiency is due to memory accesses needed for reloading or setting up the SIMD or systolic array. 

The current abstraction of v100 is a bit crude. It does not model tensor core (4x4x4 systolic array) nor it does the mem subsystem or interconnect. 
But it provides a fairly good estimation if the tensor core performance can be drilled down into an "efficiency" (from 0 to 1) over the 64 FMACs
where the efficiency approximates the time lost in setting up the state and other artifacts not modeled.

The other option is hpc1024 - a more freewheeling approximation of an all2all topology of CPUs with low-bandwidth interconnects (like 100GbE). 
No variable delays, no queueing modeling. It is an estimation on an entirely synchronized execution with no other traffic, and a lot depends on getting the
efficiency factor right so it approximates all other artifacts.

For any other configuration, just edit the setConfig routine.

The bandwidth, frequency, and fmacs (width of the simd or systolic array) can be specified. An fmac performs a multiply and an add. So, in a reduction, 
the multiply is unused.

This approximation is fairly acceptable in a cluster with uniform P2P bandwidth. For multi-bandwidth clusters, the algorithm has to be enhanced for topology and the bandwidth, but this algo is more friendlier towards dual-level networks because the reductions and exchanges both happen within a group of processors rather than any-to-any.

The code finds the partitioning, initializes the processors, and runs the algorithm that includes computation, reduction and exchanges.
A cycle count is returned for each task (computation, reduction, exchange). With the frequency, the effective TFLOPs is measured.

### Comparison with Other Algorithms

TBD. The same appartus will be used to run Cannon's and probably other algorithms like Summa in future.

### Sample output

![2688-dimension square matrix multiplication](https://github.com/bpudiped/MosaicMM/blob/master/mosaicLog2.PNG)





