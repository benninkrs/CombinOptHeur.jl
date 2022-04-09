# CombinOptHeuristics.jl
**CombinOptHeuristics** provides heuristic solvers for combinatorial optimization. Types and constructors are provided to facilitate the specification of both generic and specialized problem instances. For benchmarking purposes, a set of standard benchmark problems are also provided.

## Problem Types

Currently, solvers are provided for two types of problems: 

- **Quadratic unconstrained binary optimization**  (QUBO).  The task is to maximize or minimize 
  $$  f(x) = x^T A x + b^T x + c $$
  over $x \in \{\text{lo},\text{hi}\}^n$, where $\{\text{lo},\text{hi}\}$  is typically either $\{0,1\}$ or $\{-1,1\}$. Specialized types of QUBO include MAXCUT, MAXSAT, and Ising energy minimization. 

- **Quadratic assignment problem** (QAP). The task is to maximize or minimize
  $$ f(p) = \sum_{{i,j}=1}^n A_{i,j} B_{p_i,p_j} $$
  over all permutations $p$ of $\{1,\ldots,n\}$. A well-known subtype of  QAP is the Travelling Salesperson Problem (TSP).


## Algorithms

QUBO problems are solved by a variant of the method described in [Boros2007].  In this approach the search space is relaxed to a continuous domain. Starting from a random point in the interior of the domain, the components of a candidate solution $x$ are progressively discretized using a greedy heuristic until a local optimum is reached. This search is fast, scaling as $O(n^2)$, and has a high probability of generating a pretty good solution. In practice one typically generates a large number of candidate solutions and selects the best one.

QAP problems are solved using a novel adaptation of the Boros 2007 algorithm.  The space of permutation matrices is relaxed to the space of approximately doubly-stochastic matrices.  Starting from a random initial matrix, a candidate matrix is progressively discretized into a permutation matrix using a greedy heuristic until a local optimum is reached. The cost of generating a single candidate solution is $O(n^3)$.  Again, many candidate solutions are generated.


## Installation
In the Julia terminal, enter package mode (type `]`) and enter

```add https\\github\benninkrs\CombinOptHeuristics.jl```


## Usage

The Julia types `QUBO` and `QAP` are provided to represent instances of these two problem types. Generic problem instances can be constructed using these types, while the functions `ising`, `maxcut`, and `travsales`. are provided to more conveniently specify instances of Ising, MAXCUT, and TSP problems. A problem is solved using the generic `solve` function.

More details can be found in the in-source documentation.