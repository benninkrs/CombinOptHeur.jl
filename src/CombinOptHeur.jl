module CombinOptHeur


export solve_qubo, solve_ising, solve_maxcut, pbr_optim, read_beasley, read_biqbin_maxcut, qubo_to_ising

using DelimitedFiles
#using LinearAlgebra
#using ElasticArrays


# """
# struct QUBO{:T<:Real}

# Quadratic Unconstrained Binary Optization (QUBO) problem instance.
# """
# struct QUBO{T<:Real}
# 	Q::Array{T}
# end # module
function read_biqbin_maxcut()
	file = (@__DIR__) * "\\..\\problems\\MAXCUT\\biqbin\\G1.txt"
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
	return W
end


function read_beasley(nn::Int, i::Int)
	# Read the file
	file = "..\\problems\\QUBO\\Beasley\\bqp$(nn)_$(i).txt"
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
	return W
end



function qubo_to_ising(W)
	# Convert parameters for quadratic unconstrained binary optimization (QUBO) to Ising model parameters.
	#
	# The QUBO problem is to maximize  F(x) = x'*W*x for x in {0,1}^n
	# The Ising problem is to minimize E(s) = s'*J*s + h'*s for s in {-1,1}^n.
	# These two can be related by taking F = -E + c, s = (-1)^x, J = -W/4, h = sum(W+W',1)/4, and c = sum(W(:))/4.
	# (Note, the diagonal of J is arbitrary. We take diag(J) = 0.)
	n = size(W,1)
	J = -W/4
	J[1:n+1:n^2] .= 0
	h = dropdims(sum(W+W', dims=2), dims=2)/4
	c = sum(J[:]) + sum(h)
	return (J,h,c)
end


"""
Find an approximately maximal cut of a graph.

`(v, c) = solve_maxcut(W, N::Int` returns v∈{0,1}^n that is an approximately maximal
cut of a graph with edge weight matrix `W`.  `c` is the value of the cut.
"""
function solve_maxcut(W, niters::Int)
	n = size(W,1)
	W = copy(W)
	W[1:n:n^2] .= 0
	W .= (W+W')/2
	w = sum(W)

	c_all = zeros(niters)
	c_best = -Inf
	v_best = zeros(n)
	x = zeros(n)
	z = zeros(n)
	for iter in 1:niters
		x .= rand(n) .- 0.5
		s = pbr_optim(W, z, x, (-1, 1), <)
		e = s'*W*s
		c = (w - e)/4
		c_all[iter] = c
		if c > c_best
			v_best .= (s .> 0)
			c_best = c
		end
	end
	return (v_best, c_best)
end


"""
Approximately minimize Ising energy using pseudo-Boolean rounding (Boros 2007).

`(s, f) = solve_ising(A, b, N::Int)` returns s∈{-1,1}^n that approximately minimizes
f(s) = s'*A*s + b'*s for real matrix `A` and vector `b`.

This is method is very fast (O(n^2) time) and has a high probability of yielding a near-optimal solution.
"""
function solve_ising(A, b, N::Int)
	n = size(A, 1)
	s = Vector{Vector{Float64}}(undef, N)
	f = Vector{eltype(A)}(undef, N)
	bias = 0.3

	# Enforce properties used by pbr_optim
	Asym = (A+A')/2
	Asym[1:n+1:n^2] .= 0

	x = Vector{eltype(A)}(undef, n)

	for i = 1:N
		# Pick a random starting point
		x .= 2 .* rand(n) .- 1
		# Bias it towards the center since that tends to give slightly better results
		x .= (1-bias).*x .+ bias.*x.^3

		s[i] = pbr_optim(Asym, b, x, (-1, 1), <)

		# compute the objective value
		f[i] = s[i]'*A*s[i] + b'*s[i]

	end
	return (s, f)
end


function solve_ising(A, b, x0::Vector)
	# compute the objective value
	s = pbr_optim(A, b, x0, (-1, 1), <)
	f = s'*A*s + b'*s

	return (s, f)
end


"""
Approximately maximize a Quadratic Binary function using pseudo-Boolean rounding (Boros 2007).

`(v, f) = solve_qubo(W, N::Int)` returns v∈{0,1}^n that approximately maximizes
f(v) = v'*W*v for real matrix `W`.

This method is very fast (O(n^2) time) and has a high probability of yielding a near-optimal solution.
"""
function solve_qubo(W, N::Int)
	n = size(W, 1)
	v = Vector{Vector{Int}}(undef, N)
	f = Vector{eltype(W)}(undef, N)
	bias = 0.3

	# Enforce properties used by pbr_optim
	A = (W+W')/2
	b = A[1:n+1:n^2]
	A[1:n+1:n^2] .= 0

	x = Vector{eltype(A)}(undef, n)

	for i = 1:N
		# Pick a random starting point
		x .= 2 .* rand(n) .- 1
		# Bias it towards the center since that tends to give slightly better results
		x .= (1-bias).*x .+ bias.*x.^3
		x .= (x .+ 1)./2

		v[i] = pbr_optim(A, b, x, (0, 1), >)

		# compute the objective value
		f[i] = v[i]'*W*v[i]

	end
	return (v, f)
end


# E. Boros et al., "Local search heuristics for Quadratic Unconstrained Binary Optimization (QUBO)",
# J. Heuristics 13, 99-132 (2007).

"""
Binary quadratic optimization (generic version) using pseudo-Boolean rounding.

Given matrix A and vector b, the goal is to find x∈{lo,hi}^n that minimizes or maximizes
f(x) = x'*A*x + b'*x.  This is accomplished via

`x = pbr_optim(A, b, x0, (lo, hi), comparator)`

where `x0` is the initial guess and `comparator` is `>`  for maximimzation and `<` for minimization.

Notes:
* All inputs should be real
* A must be symmetric and 0 on the diagonal
"""
function pbr_optim(A::AbstractMatrix, b::AbstractVector, x0::AbstractVector, (lo, hi), isbetter)
	# A must be symmetric and zero on the diagonal.
	# This ensures the maximum of f is at a corner of the hypercube, i.e. a binary solution.

	x = copy(x0)
	n = length(x)

	# the gradient at x
	g = 2*(A*x) + b

	local xc::Float64			# candidate updated coordinate
	local xc_upd::Float64	# value of best updated coordinate
	local df_best::Float64

	while true
		i_upd = 1
		for ic = 1:n
			# determine the optimal value of coordinate i
			xc = isbetter(g[ic], 0) ? Float64(hi) : Float64(lo)
			# determine the change in f
			df = (xc - x[ic]) * g[ic]
			# remember which coordinate yielded the most improvement
			if ic == 1 || isbetter(df, df_best)
				df_best = df
				i_upd = ic
				xc_upd = xc
			end
		end


		if xc_upd != x[i_upd]
			# A change in x is prescribed.  Update the gradient
			#g .+= A[:, i_upd] .* (2*(xc_upd - x[i_upd]))
			for i = 1:n
				g[i] += A[i, i_upd] * 2*(xc_upd - x[i_upd])
			end
			# update x
			x[i_upd] = xc_upd
		else
			# The best change is no change at all;  quit.
			break
		end
	end

	# Round any remaining fractional components (their derivatives could have been 0) and cast as integer
	return [(y>0 ? hi : lo) for y in x]
end


end