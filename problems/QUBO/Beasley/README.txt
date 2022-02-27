The problems in this directory are known as the "Beasley set" and were obtained from http://people.brunel.ac.uk/~mastjjb/jeb/orlib/bqpinfo.html

(Below is an excerpt from that website):
--------------------------------------

There are currently 7 data files:
bqpgka, bqp50, bqp100, bqp250, bqp500, bqp1000, bqp2500

These data files are the test problems used in the working paper: 
"Heuristic algorithms for the unconstrained binary quadratic programming 
problem" by J.E. Beasley available from here.

The problem as given in the data files below is to maximise the expression
     sum{i=1,...,n} sum{j=1,...,n} q(i,j)x(i)x(j)
where n is the number of variables and q(i,j) is a symmetric matrix.
The x(i) {i=1,...,n} are the binary (zero-one) variables.

The format of these data files is:
number of test problems
for each test problem in turn:
   number of variables (n), number of non-zero elements in the q(i,j) matrix
       for each non-zero element in turn: 
       i, j, q(i,j) {=q(j,i) as the matrix is symmetric}