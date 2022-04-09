"""
Heuristic solvers for cominbatorial optimization problems, in particular:
  • Quadratic unconstrained binary optimization (QUBO)
  • Quadratic assignment (QAP)

This includes the following as special cases:  Ising energy minimization, MAXCUT, MAXSAT, and travelling salesman.
Many other combinatorial problems can be reduced to one of these two.
"""
module CombinOptHeuristics


#export solve_qubo, solve_ising, solve_maxcut, pbr_optim, read_beasley, read_biqbin_maxcut, qubo_to_ising
export QUBO, ising, maxcut, solve, solve_qubo
export read_beasley, read_biqbin_maxcut


# import Base.size


using DelimitedFiles
#using LinearAlgebra
#using ElasticArrays


"""
General quadratic unconstrained binary optimization (QUBO) problem

F(x) = x'*A*x + b'*x + c
(min || max) F(x)  for x ∈ {lo,hi}^n

QUBO
	A::Matrix{Float64}
	b::Vector{Float64}
	c::Float64
	values::Tuple{Float64,Float64}
	mode::Function							min or max
"""
struct QUBO
	A::Matrix{Float64}
	b::Vector{Float64}
	c::Float64
	values::Tuple{Float64, Float64}
	mode::Function
	function QUBO(A, b, c, (lo,hi), f)
		n = length(b)
		if size(A) != (n,n)
			error("A and b are incompatible sizes")
		end
		if lo>=hi
			error("lo mst be less than hi")
		end
		if !(f == min || f === max)
			error("mode must be min or max")
		end

		new(A,b,c, (lo,hi), f)
	end
end

# Convenience constructors
QUBO(A, lohi::Tuple{<:Real, <:Real}, mode::Function) = QUBO(A, zeros(size(A,1)), 0, lohi, mode)
QUBO(A, b, lohi::Tuple{<:Real, <:Real}, mode::Function) = QUBO(A, b, 0, lohi, mode)

nvars(q::QUBO) = length(q.b)

"""
Evaluate a QUBO at x
"""
(Q::QUBO)(x) = x'*Q.A*x + Q.b'*x + Q.c


"""
Create an Ising problem

`ising(J,h)` creates QUBO object representing the following problem:

   minimize s'*J*s + h'*s + c   over  s∈{-1,1}^n
"""
function ising(J, h, c = 0)
	QUBO(J, h, 0, (-1,1), min)
end
ising(J) = ising(J, zeros(size(J,1)))



"""
Create a MAXCUT problem

`maxcut(J,h)` creates QUBO object representing the following problem:

   maximize sum_i,j W[i,j] * x[j] * (1-x[j])   over  x∈{0,1}^n

where `W` is the symmetric edge weight matrix.
"""
function maxcut(W)
	QUBO(-W, dropdims(sum(W; dims=2); dims=2), 0, (0,1), max)		# wrong
end


"""
`read_beasley(n, i)` returns the Beasley QUBO problem with `n` variables, instance `i`.
`n`∈{50,100,250,500,1000,2500,5000} and `i`∈{1,...,10}.
"""
read_beasley(nn::Int, i::Int) = read_beasley("bqp$(nn)_$(i)")

"""
`read_biqbin_maxcut(name)` returns the Beasley QUBO problem from the named file (without extension).
"""

function read_beasley(name)
	file =  (@__DIR__) * "\\..\\problems\\QUBO\\Beasley\\$(name).txt"
	@info "reading $file"
	IJV = readdlm(file)

	# populate the QBO matrix
	n = round(Int, IJV[1,1])
	W = zeros(n,n)
	for k = 2:size(IJV,1)
		i = round(Int, IJV[k, 1])
		j = round(Int, IJV[k, 2])
		v = IJV[k,3]
		W[i,j] = v
		W[j,i] = v
	end

	return QUBO(W, (0,1), max)
end


"""
`read_biqbin_maxcut(name)` returns BiqBin MAXCUT problem from the named file (without extension).
"""

function read_biqbin_maxcut(name)
	file = (@__DIR__) * "\\..\\problems\\MAXCUT\\biqbin\\" * name * ".txt"
	@info "reading $file"
	(IJW, header) = readdlm(file; header=true)
	nv = parse(Int, header[1])
	ne = parse(Int, header[2])

	# populate the QBO matrix
	W = zeros(nv,nv)
	for ie = 1:ne
		i = round(Int, IJW[ie, 1])
		j = round(Int, IJW[ie, 2])
		w = IJW[ie,3]
		W[i,j] = w
		W[j,i] = w
	end
	return maxcut(W)
end



"""
Solve a QUBO problem instance

`(f(x),x) = solve(q::QUBO, N)` finds x such that f(x) is (with high probability nearly) the optimal value of `q`.
`N` is the number of candidate solutions to consider; only the best is returned.
"""
function solve(q::QUBO, niters::Integer)
	n = nvars(q)
	lo = q.values[1]
	hi = q.values[2]
	A = (q.A + q.A')/2
	diag = 1:n+1:n^2
	a = A[diag]
	b = q.b + (hi + lo)*a
	A[diag] .= 0

	isbetter = (q.mode === max) ? (>) : (<)

	f_best = NaN
	x_best = zeros(n)
	for iter in 1:niters
		x = solve_qubo(A, b, q.values, isbetter)
		f = q(x)
		if iter == 1 || isbetter(f, f_best)
			x_best = x
			f_best = f
		end
	end
	(f_best, x_best)
end



"""
Core solver for QUBO.  Typically call solve(::QUBO) instead.
"""
function solve_qubo(A::AbstractMatrix, b::AbstractVector, (lo, hi), isbetter, x::AbstractVector)
	# A must be symmetric and zero on the diagonal.
	# This ensures the maximum of f is at a corner of the hypercube, i.e. a binary solution.

	n = length(x)

	# the gradient at x
	g = 2*(A*x) + b

	x_upd = NaN					# value of best updated coordinate
	df_best = NaN

	while true
		i_upd = 1
		for ic = 1:n
			# determine the optimal value of coordinate i
			if iszero(g[ic])
				x_ = rand()>0.5 ? Float64(hi) : Float64(lo)
			else
				x_ = isbetter(g[ic], 0) ? Float64(hi) : Float64(lo)
			end
			# determine the change in f
			df = (x_ - x[ic]) * g[ic]
			# remember which coordinate yielded the most improvement
			if ic == 1 || isbetter(df, df_best)
				df_best = df
				i_upd = ic
				x_upd = x_
			end
		end


		if x_upd != x[i_upd]
			# A change in x is prescribed.  Update the gradient
			for i = 1:n
				g[i] += A[i, i_upd] * 2*(x_upd - x[i_upd])
			end
			# update x
			x[i_upd] = x_upd
		else
			# The best change is no change at all;  quit.
			break
		end
	end

	# Round any remaining fractional components (their derivatives could have been 0) and cast as integer
	mid = (hi+lo)/2
	for i in eachindex(x)
		if x[i] == mid
			x[i] = rand()>0 ? hi : lo
		elseif x[i] > mid
			x[i] = hi
		else
			x[i] = lo
		end
	end
	#  [(y>mid ? hi : lo) for y in x]
	return x
end


# Solve the QUBO using a random starting point
function solve_qubo(A, b, (lo, hi), isbetter)
	x = lo .+ (hi-lo)*(0.25 .+ rand(length(b))/2.0)
	solve_qubo(A, b, (lo, hi), isbetter, x)
end




# function qubo_to_ising(W)
# 	# Convert parameters for quadratic unconstrained binary optimization (QUBO) to Ising model parameters.
# 	#
# 	# The QUBO problem is to maximize  F(x) = x'*W*x for x in {0,1}^n
# 	# The Ising problem is to minimize E(s) = s'*J*s + h'*s for s in {-1,1}^n.
# 	# These two can be related by taking F = -E + c, s = (-1)^x, J = -W/4, h = sum(W+W',1)/4, and c = sum(W(:))/4.
# 	# (Note, the diagonal of J is arbitrary. We take diag(J) = 0.)
# 	n = size(W,1)
# 	J = -W/4
# 	J[1:n+1:n^2] .= 0
# 	h = dropdims(sum(W+W', dims=2), dims=2)/4
# 	c = sum(J[:]) + sum(h)
# 	return (J,h,c)
# end


end