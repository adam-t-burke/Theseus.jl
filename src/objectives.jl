"""
Softplus is a smooth approximation of the ReLU function.
A sharpness parameter k may be included to make the inflection point more precise.

x is the input, b is the inflection point bias, k is the sharpness parameter for the barrier slope.

negative k raises a barrier on the left side of the inflection point.
positive k raises a barrier on the right side of the inflection point.
"""
@inline function softplus(x::Float64, b::Float64, k::Float64)
    log1p(exp(-k * (b - x) - 1))
end


"""
Penalizes values in vector that are below a threshold 
"""
function minPenalty(x::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64}, k::Float64)
    total = 0.0
    @inbounds for (pos, idx) in enumerate(indices)
        total += softplus(x[idx], values[pos], -k)
    end
    total
end

function minPenalty(x::Vector{Float64}, values::Vector{Float64}, k::Float64)
    total = 0.0
    @inbounds for i in eachindex(x, values)
        total += softplus(x[i], values[i], -k)
    end
    total
end

"""
Penalizes values in vector that are above a threshold 
"""
function maxPenalty(x::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64}, k::Float64)
    total = 0.0
    @inbounds for (pos, idx) in enumerate(indices)
        total += softplus(x[idx], values[pos], k)
    end
    total
end

function maxPenalty(x::Vector{Float64}, values::Vector{Float64}, k::Float64)
    total = 0.0
    @inbounds for i in eachindex(x, values)
        total += softplus(x[i], values[i], k)
    end
    total
end

"""
Penalize values to be between lb and ub with a smooth approximation of ReLU. 
Prevents discontinuities in the objective function.
"""

function pBounds(p::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, kl::Float64, ku::Float64)
    return minPenalty(p, lb, kl) + maxPenalty(p, ub, ku)
end
"""
Minimze distances between selected target nodes and their corresponding nodes in the form found network.
"""
function target_xyz(xyz::Matrix{Float64}, target::Matrix{Float64}, indices::Vector{Int})
    total = 0.0
    cols = size(target, 2)
    @inbounds for (pos, idx) in enumerate(indices)
        for col in 1:cols
            diff = xyz[idx, col] - target[pos, col]
            total += diff * diff
        end
    end
    total
end

"""
Minimize the distance between the x and y coordinates of the target nodes and their corresponding nodes in the form found network.
Equal to targeting a plan projection of the target nodes. Useful if the target geometry variation is dominated by the x and y coordinates.
"""
function target_xy(xyz::Matrix{Float64}, target::Matrix{Float64}, indices::Vector{Int})
    total = 0.0
    @inbounds for (pos, idx) in enumerate(indices)
        for col in 1:2
            diff = xyz[idx, col] - target[pos, col]
            total += diff * diff
        end
    end
    total
end

"""
Find the distance between all pairs of points in a point set. Returns a strictly lower triangular matrix.
"""
function pairDist(xyz)
    n = size(xyz, 1)
    # Create the distance matrix without mutation
    [i > j ? norm(xyz[i,:] - xyz[j,:]) : 0.0 for i in 1:n, j in 1:n]
end

"""
Compare the distance between all pairs of points in a target point set and the distance between all pairs of points in a form found point set.
"""
function rigidSetCompare(xyz, indices, target)
    xyz = xyz[indices,:]
    test_distances = pairDist(xyz)
    target_distances = pairDist(target)
    return sum((target_distances - test_distances).^2)
end

"""
Compute difference between the maximum and minimum lengths of the edges in the network.
"""
function lenVar(x::Vector{Float64}, indices::Vector{Int64})
    min_val = Inf
    max_val = -Inf
    @inbounds for idx in indices
        val = x[idx]
        if val < min_val
            min_val = val
        end
        if val > max_val
            max_val = val
        end
    end
    max_val - min_val
end

"""
Reduce the difference between the maximum and minimum forces in the network.
From Schek theorem 2. 
"""
function forceVar(x::Vector{Float64}, indices::Vector{Int64})
    min_val = Inf
    max_val = -Inf
    @inbounds for idx in indices
        val = x[idx]
        if val < min_val
            min_val = val
        end
        if val > max_val
            max_val = val
        end
    end
    max_val - min_val
end

"""
Minimize the difference between the form found lengths of the edges and the target lengths.
"""

function lenTarget(lengths::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64})
    total = 0.0
    @inbounds for (pos, idx) in enumerate(indices)
        diff = lengths[idx] - values[pos]
        total += diff * diff
    end
    total
end



"""
Cross entropy loss function.
"""
function crossEntropyLoss(t::Vector{Float64}, p::Vector{Float64})
    -sum(t .* log1p.(p))
end

"""
Weighted cross entropy loss, best used with a plain mask vector of weights.
"""
function crossEntropyLoss(t::Vector{Float64}, p::Vector{Float64}, w::Union{SparseVector{Float64, Int64}, Vector{Float64}})
    -vec(t .* log1p.(p))' * w
end

"""
The derivative of the softplus function is the logistic function.
A scaling parameter k can be introduced to make the inflection point more precise.
This is a smooth approximation of the heaviside step function. 
https://en.wikipedia.org/wiki/Logistic_function
"""
function logisticFunc(x)
    1 / (1 + exp.(-x))
end

function logisticFunc(x, k::Union{Float64,Int64})
    1 / (1 + exp.(-k*x))
end

"""
LogSumExp is the multivariable generalization of the logistic function.
"""
function logSumExp(x)
    log1p(sum(exp.(x)))
end

"""
Softmax is a generalized version of the logistic function.
It returns a vector of probabilities that sum to 1.
"""
function softmax(x)
    exp.(x) / sum(exp.(x))
end

function softmin(x)
    softmax(-x)
end


