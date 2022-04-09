# CombinOptHeuristics.jl
**CombinOptHeuristics** provides heuristic solvers for combinatorial optimization. It additionally provides types and constructors to for problem instances and some standard benchmark problems.

- The types `QUBO` and `QAP` respectively represent quadratic unconstrained binary optimization (QUBO) problems and quadratic assignment problems (QAP).
- Convenience functions `ising`, `maxcut`, and `travsales` facilitate the construction of Ising energy minimization, MAXCUT, and travelling salesperson problems.
- The `solve` function generates heuristic solutions for a given problem.

More details can be found in the source documentation.



## Installation
In the Julia terminal, enter package mode (type `]`) and enter

```add https://github.com/benninkrs/CombinOptHeuristics.jl```


## Example Usage
This example shows how to construct and solve a simple QUBO problem involving 3 variables: 
```
julia> using CombinOptHeuristics

# create a maximization QUBO problem over {0,1}^3
julia> A = [0 3 -4; 3 0 6; -4 6 0]; b = [0.2, -0.7, 0]
julia> q = QUBO(A, b, 0, (0,1), max);

# evaluate the objective function at [1,0,1]
julia> q([1,0,1])                       
-7.8

# find a good solution
julia> (f,x) = solve(q, 100)            
(11.3, [0.0, 1.0, 1.0])

julia> q(x)                             
11.3
```

This example shows how to solve a benchmark problem:
```
# read a MAXCUT problem from BiqBin
julia> q = read_biqbin_maxcut("G1");    

julia> (f,x) = solve(q,100)
(11540.0, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0  …  0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
```
Because the heuristic is randomized, the results may be slightly different.

## Technical Details

### Problem Types

Currently, solvers are provided for two types of problems: 

- **Quadratic unconstrained binary optimization**  (QUBO).  The task is to maximize or minimize 
  $$  f(x) = x^T A x + b^T x + c $$
  over $x \in \{\text{lo},\text{hi}\}^n$, where $\{\text{lo},\text{hi}\}$  is typically either $\{0,1\}$ or $\{-1,1\}$. Specialized types of QUBO include MAXCUT, MAXSAT, and Ising energy minimization. 

- **Quadratic assignment problem** (QAP). The task is to maximize or minimize
  $$ f(p) = \sum_{{i,j}=1}^n A_{i,j} B_{p_i,p_j} $$
  over all permutations $p$ of $\{1,\ldots,n\}$. A well-known subtype of  QAP is the Travelling Salesperson Problem (TSP).


### Algorithms

QUBO problems are solved by a variant of the method described in [Boros2007].  In this approach the search space is relaxed to a continuous domain. Starting from a random point in the interior of the domain, the components of a candidate solution $x$ are progressively discretized using a greedy heuristic until a local optimum is reached. This search is fast, scaling as $O(n^2)$, and has a high probability of generating a pretty good solution. In practice one typically generates a large number of candidate solutions and selects the best one.

QAP problems are solved using a novel adaptation of the Boros 2007 algorithm.  The space of permutation matrices is relaxed to the space of approximately doubly-stochastic matrices.  Starting from a random initial matrix, a candidate matrix is progressively discretized into a permutation matrix using a greedy heuristic until a local optimum is reached. The cost of generating a single candidate solution is $O(n^3)$.  Again, many candidate solutions are generated.