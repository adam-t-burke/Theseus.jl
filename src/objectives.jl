"""
Softplus is a smooth approximation of the ReLU function.
A sharpness parameter k may be included to make the inflection point more precise.

x is the input, b is the inflection point bias, k is the sharpness parameter for the barrier slope.

negative k raises a barrier on the left side of the inflection point.
positive k raises a barrier on the right side of the inflection point.
"""
const _has_log1pexp = isdefined(Base.Math, :log1pexp)

@inline function _log1pexp(z)
    if _has_log1pexp
        return Base.Math.log1pexp(z)
    elseif z > zero(z)
        return z + log1p(exp(-z))
    else
        return log1p(exp(z))
    end
end

@inline function softplus(x::T, b::S, k::R) where {T<:Real,S<:Real,R<:Real}
    z = -k * (b - x) - oneunit(k)
    return _log1pexp(z)
end

softplus(x::AbstractVector{T}, b::AbstractVector{S}, k::R) where {T<:Real,S<:Real,R<:Real} = softplus.(x, b, Ref(k))


"""
Penalizes values in vector that are below a threshold 
"""
function minPenalty(x::AbstractVector{<:Real}, values::AbstractVector{<:Real}, indices::AbstractVector{<:Integer}, k::Real)
    loss = zero(eltype(x))
    for (i, idx) in enumerate(indices)
        v = values[i]
        if isfinite(v)
            loss += softplus(x[idx], v, -k)
        end
    end
    loss
end

function minPenalty(x::AbstractVector{<:Real}, values::AbstractVector{<:Real}, k::Real)
    loss = zero(eltype(x))
    for i in 1:length(x)
        v = values[i]
        if isfinite(v)
            loss += softplus(x[i], v, -k)
        end
    end
    loss
end

"""
Penalizes values in vector that are above a threshold 
"""
function maxPenalty(x::AbstractVector{<:Real}, values::AbstractVector{<:Real}, indices::AbstractVector{<:Integer}, k::Real)
    loss = zero(eltype(x))
    for (i, idx) in enumerate(indices)
        v = values[i]
        if isfinite(v)
            loss += softplus(x[idx], v, k)
        end
    end
    loss
end

function maxPenalty(x::AbstractVector{<:Real}, values::AbstractVector{<:Real}, k::Real)
    loss = zero(eltype(x))
    for i in 1:length(x)
        v = values[i]
        if isfinite(v)
            loss += softplus(x[i], v, k)
        end
    end
    loss
end

"""
Penalize values to be between lb and ub with a smooth approximation of ReLU. 
Prevents discontinuities in the objective function.
"""

function pBounds(p::AbstractVector{<:Real}, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}, kl::Real, ku::Real)
    minPenalty(p, lb, kl) + maxPenalty(p, ub, ku)
end

function pBounds(p::AbstractVector{<:Real}, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}, lb_indices::AbstractVector{<:Integer}, ub_indices::AbstractVector{<:Integer}, kl::Real, ku::Real)
    loss = zero(eltype(p))
    for idx in lb_indices
        v = lb[idx]
        if isfinite(v)
            loss += softplus(p[idx], v, -kl)
        end
    end
    for idx in ub_indices
        v = ub[idx]
        if isfinite(v)
            loss += softplus(p[idx], v, ku)
        end
    end
    loss
end

"""
Minimze distances between selected target nodes and their corresponding nodes in the form found network.
"""
function target_xyz(xyz::AbstractMatrix{<:Real}, target::AbstractMatrix{<:Real}, indices::AbstractVector{<:Integer})
    loss = zero(eltype(xyz))
    for (i, idx) in enumerate(indices)
        loss += (xyz[idx, 1] - target[i, 1])^2
        loss += (xyz[idx, 2] - target[i, 2])^2
        loss += (xyz[idx, 3] - target[i, 3])^2
    end
    loss
end

"""
Minimize the distance between the x and y coordinates of the target nodes and their corresponding nodes in the form found network.
Equal to targeting a plan projection of the target nodes. Useful if the target geometry variation is dominated by the x and y coordinates.
"""
function target_xy(xyz::AbstractMatrix{<:Real}, target::AbstractMatrix{<:Real}, indices::AbstractVector{<:Integer})
    loss = zero(eltype(xyz))
    for (i, idx) in enumerate(indices)
        loss += (xyz[idx, 1] - target[i, 1])^2
        loss += (xyz[idx, 2] - target[i, 2])^2
    end
    loss
end

"""
Find the distance between all pairs of points in a point set. Returns a strictly lower triangular matrix.
"""
function pairDist(xyz::AbstractMatrix{<:Real})
    n = size(xyz, 1)
    T = eltype(xyz)
    dist = zeros(T, n, n)
    for i in 1:n
        for j in 1:(i-1)
            dx = xyz[i, 1] - xyz[j, 1]
            dy = xyz[i, 2] - xyz[j, 2]
            dz = xyz[i, 3] - xyz[j, 3]
            dist[i, j] = sqrt(dx^2 + dy^2 + dz^2)
        end
    end
    return dist
end

"""
Compare the distance between all pairs of points in a target point set and the distance between all pairs of points in a form found point set.
"""
function rigidSetCompare(xyz::AbstractMatrix{<:Real}, indices::AbstractVector{<:Integer}, target::AbstractMatrix{<:Real})
    loss = zero(eltype(xyz))
    n = length(indices)
    for i in 1:n
        idx_i = indices[i]
        for j in 1:(i-1)
            idx_j = indices[j]
            
            # Distance in network
            dx = xyz[idx_i, 1] - xyz[idx_j, 1]
            dy = xyz[idx_i, 2] - xyz[idx_j, 2]
            dz = xyz[idx_i, 3] - xyz[idx_j, 3]
            d_network = sqrt(dx^2 + dy^2 + dz^2)
            
            # Distance in target
            tx = target[i, 1] - target[j, 1]
            ty = target[i, 2] - target[j, 2]
            tz = target[i, 3] - target[j, 3]
            d_target = sqrt(tx^2 + ty^2 + tz^2)
            
            loss += (d_target - d_network)^2
        end
    end
    return loss
end

"""
Compute difference between the maximum and minimum lengths of the edges in the network.
"""
function lenVar(x::AbstractVector{<:Real}, indices::AbstractVector{<:Integer})
    if isempty(indices); return zero(eltype(x)); end
    v_min = x[indices[1]]
    v_max = v_min
    for i in 2:length(indices)
        v = x[indices[i]]
        if v < v_min; v_min = v; end
        if v > v_max; v_max = v; end
    end
    v_max - v_min
end

"""
Reduce the difference between the maximum and minimum forces in the network.
From Schek theorem 2. 
"""
function forceVar(x::AbstractVector{<:Real}, indices::AbstractVector{<:Integer})
    if isempty(indices); return zero(eltype(x)); end
    v_min = x[indices[1]]
    v_max = v_min
    for i in 2:length(indices)
        v = x[indices[i]]
        if v < v_min; v_min = v; end
        if v > v_max; v_max = v; end
    end
    v_max - v_min
end

"""
Minimize the difference between the form found lengths of the edges and the target lengths.
"""

function lenTarget(lengths::AbstractVector{<:Real}, values::AbstractVector{<:Real}, indices::AbstractVector{<:Integer})
    loss = zero(eltype(lengths))
    for (i, idx) in enumerate(indices)
        loss += (lengths[idx] - values[i])^2
    end
    loss
end


"""
    anchor_reactions(topology, q, xyz)

Compute reaction force vectors for every node in the network. Returns an
`nn × 3` matrix whose rows align with the global node indexing used across
the problem definition. Rows associated with free nodes are zero.
"""
function anchor_reactions(topo::NetworkTopology, q::AbstractVector{<:Real}, xyz::AbstractMatrix{<:Real})
    @assert size(xyz, 1) == topo.num_nodes "Geometry matrix must include all nodes"
    edge_vectors = topo.incidence * xyz
    axial_vectors = edge_vectors .* q
    fixed_reactions = -topo.fixed_incidence' * axial_vectors

    n_free = length(topo.free_node_indices)
    n_fixed = length(topo.fixed_node_indices)
    dim = size(xyz, 2)

    T = promote_type(eltype(q), eltype(xyz))

    if n_fixed == 0
        return zeros(T, topo.num_nodes, dim)
    end

    fixed_block = eltype(fixed_reactions) === T ? fixed_reactions : convert(Matrix{T}, fixed_reactions)

    if n_free == 0
        return fixed_block
    else
        free_block = zeros(T, n_free, dim)
        return vcat(free_block, fixed_block)
    end
end

function anchor_reactions!(ctx::FDMContext, topo::NetworkTopology, q::AbstractVector{<:Real})
    n_free = length(topo.free_node_indices)
    ctx.reactions .= 0.0
    # Reactions on fixed nodes: -Cf' * diag(q) * member_vectors
    # Diagonal(q) * ctx.member_vectors scales rows of member_vectors by q
    ctx.reactions[n_free+1:end, :] .= -topo.fixed_incidence' * (Diagonal(q) * ctx.member_vectors)
end

"""
    reaction_direction_misalignment(reaction, target_dir)

Penalty measuring angular deviation between a reaction vector and a unit
target direction. Returns zero when the vectors align and increases towards
two when they oppose each other. A zero reaction incurs a full penalty.
"""
function reaction_direction_misalignment(reaction::AbstractVector{<:Real}, target_dir::AbstractVector{<:Real})
    r_norm = norm(reaction)
    unit_val = oneunit(r_norm)
    if iszero(r_norm)
        return unit_val
    end
    dot_dir = clamp(dot(reaction, target_dir) / r_norm, -unit_val, unit_val)
    unit_val - dot_dir
end

function reaction_direction_loss(reactions::AbstractMatrix{<:Real}, objective::ReactionDirectionObjective)
    total = zero(eltype(reactions))
    for (row_idx, node_idx) in enumerate(objective.anchor_indices)
        reaction = @view reactions[node_idx, :]
        target_dir = @view objective.target_directions[row_idx, :]
        total += reaction_direction_misalignment(reaction, target_dir)
    end
    total
end

function reaction_direction_magnitude_loss(reactions::AbstractMatrix{<:Real}, objective::ReactionDirectionMagnitudeObjective)
    total = zero(eltype(reactions))
    zero_val = zero(eltype(reactions))
    for (row_idx, node_idx) in enumerate(objective.anchor_indices)
        reaction = @view reactions[node_idx, :]
        target_dir = @view objective.target_directions[row_idx, :]
        dir_loss = reaction_direction_misalignment(reaction, target_dir)
        mag_loss = max(norm(reaction) - objective.target_magnitudes[row_idx], zero_val)
        total += dir_loss + mag_loss
    end
    total
end



"""
Cross entropy loss function.
"""
function crossEntropyLoss(t::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    -sum(t .* log1p.(p))
end

"""
Weighted cross entropy loss, best used with a plain mask vector of weights.
"""
function crossEntropyLoss(t::AbstractVector{<:Real}, p::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    -vec(t .* log1p.(p))' * w
end

function crossEntropyLoss(t::AbstractVector{<:Real}, p::AbstractVector{<:Real}, w::SparseVector{T, Ti}) where {T<:Real, Ti<:Integer}
    -vec(t .* log1p.(p))' * w
end

"""
The derivative of the softplus function is the logistic function.
A scaling parameter k can be introduced to make the inflection point more precise.
This is a smooth approximation of the heaviside step function. 
https://en.wikipedia.org/wiki/Logistic_function
"""
function logisticFunc(x::Real)
    one = oneunit(x)
    one / (one + exp(-x))
end

function logisticFunc(x::AbstractArray{<:Real})
    one = oneunit(eltype(x))
    one ./ (one .+ exp.(-x))
end

function logisticFunc(x::Real, k::Real)
    one = oneunit(x)
    one / (one + exp(-k * x))
end

function logisticFunc(x::AbstractArray{<:Real}, k::Real)
    one = oneunit(eltype(x))
    one ./ (one .+ exp.(-k .* x))
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
    exp_x = exp.(x)
    exp_x ./ sum(exp_x)
end

softmin(x) = softmax(-x)

