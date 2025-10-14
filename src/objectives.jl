"""
Softplus is a smooth approximation of the ReLU function.
A sharpness parameter k may be included to make the inflection point more precise.

x is the input, b is the inflection point bias, k is the sharpness parameter for the barrier slope.

negative k raises a barrier on the left side of the inflection point.
positive k raises a barrier on the right side of the inflection point.
"""
@inline function softplus(x::Float64, b::Float64, k::Float64)
    z = -k * (b - x) - 1
    if z > 0
        return z + log1p(exp(-z))
    else
        return log1p(exp(z))
    end
end

softplus(x::Vector{Float64}, b::Vector{Float64}, k::Float64) = softplus.(x, b, Ref(k))


"""
Penalizes values in vector that are below a threshold 
"""
function minPenalty(x::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64}, k::Float64)
    selected = x[indices]
    mask = isfinite.(values)
    if !any(mask)
        return 0.0
    end
    sum(softplus(selected[mask], values[mask], -k))
end

function minPenalty(x::Vector{Float64}, values::Vector{Float64}, k::Float64)
    mask = isfinite.(values)
    if !any(mask)
        return 0.0
    end
    sum(softplus(x[mask], values[mask], -k))
end

"""
Penalizes values in vector that are above a threshold 
"""
function maxPenalty(x::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64}, k::Float64)
    selected = x[indices]
    mask = isfinite.(values)
    if !any(mask)
        return 0.0
    end
    sum(softplus(selected[mask], values[mask], k))
end

function maxPenalty(x::Vector{Float64}, values::Vector{Float64}, k::Float64)
    mask = isfinite.(values)
    if !any(mask)
        return 0.0
    end
    sum(softplus(x[mask], values[mask], k))
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
function target_xyz(xyz, target, indices)
    sum((xyz[indices,:] - target).^2)
end

"""
Minimize the distance between the x and y coordinates of the target nodes and their corresponding nodes in the form found network.
Equal to targeting a plan projection of the target nodes. Useful if the target geometry variation is dominated by the x and y coordinates.
"""
function target_xy(xyz, target, indices)
    sum((xyz[indices,1:2] - target[:,1:2]).^2)
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
    x = x[indices]
    -reduce(-, extrema(x))
end

"""
Reduce the difference between the maximum and minimum forces in the network.
From Schek theorem 2. 
"""
function forceVar(x::Vector{Float64}, indices::Vector{Int64})
    x = x[indices]
    -reduce(-, extrema(x))
end

"""
Minimize the difference between the form found lengths of the edges and the target lengths.
"""

function lenTarget(lengths::Vector{Float64}, values::Vector{Float64}, indices::Vector{Int64})
    sum((lengths[indices] - values).^2)
end


"""
    anchor_reactions(topology, q, xyz)

Compute reaction force vectors for every node in the network. Returns an
`nn × 3` matrix whose rows align with the global node indexing used across
the problem definition. Rows associated with free nodes are zero.
"""
function anchor_reactions(topo::NetworkTopology, q::AbstractVector{<:Real}, xyz::Matrix{Float64})
    @assert size(xyz, 1) == topo.num_nodes "Geometry matrix must include all nodes"
    edge_vectors = topo.incidence * xyz
    axial_vectors = edge_vectors .* q
    fixed_reactions = -topo.fixed_incidence' * axial_vectors

    n_free = length(topo.free_node_indices)
    n_fixed = length(topo.fixed_node_indices)
    dim = size(xyz, 2)

    if n_fixed == 0
        return zeros(Float64, topo.num_nodes, dim)
    elseif n_free == 0
        return fixed_reactions
    else
        free_block = zeros(Float64, n_free, dim)
        return vcat(free_block, fixed_reactions)
    end
end

"""
    reaction_direction_misalignment(reaction, target_dir)

Penalty measuring angular deviation between a reaction vector and a unit
target direction. Returns zero when the vectors align and increases towards
two when they oppose each other. A zero reaction incurs a full penalty.
"""
function reaction_direction_misalignment(reaction::AbstractVector{<:Real}, target_dir::AbstractVector{<:Real})
    r_norm = norm(reaction)
    if r_norm <= eps(Float64)
        return 1.0
    end
    dot_dir = clamp(dot(reaction, target_dir) / r_norm, -1.0, 1.0)
    1.0 - dot_dir
end

function reaction_direction_loss(reactions::Matrix{Float64}, objective::ReactionDirectionObjective)
    total = 0.0
    for (row_idx, node_idx) in enumerate(objective.anchor_indices)
        reaction = @view reactions[node_idx, :]
        target_dir = @view objective.target_directions[row_idx, :]
        total += reaction_direction_misalignment(reaction, target_dir)
    end
    total
end

function reaction_direction_magnitude_loss(reactions::Matrix{Float64}, objective::ReactionDirectionMagnitudeObjective)
    total = 0.0
    for (row_idx, node_idx) in enumerate(objective.anchor_indices)
        reaction = @view reactions[node_idx, :]
        target_dir = @view objective.target_directions[row_idx, :]
        dir_loss = reaction_direction_misalignment(reaction, target_dir)
        mag_loss = max(norm(reaction) - objective.target_magnitudes[row_idx], 0.0)
        total += dir_loss + mag_loss
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


