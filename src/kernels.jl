using CUDA

"""
    kernel_compute_geometry!(x, edge_nodes, q, L, F)

Compute edge lengths and member forces on the GPU.
`x`: (N, 3) nodal positions.
`edge_nodes`: (M, 2) edge-node connectivity.
`q`: (M, 1) force densities.
`L`: (M, 1) output lengths.
`F`: (M, 1) output forces.
"""
function kernel_compute_geometry!(x, edge_nodes, q, L, F)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= size(edge_nodes, 1)
        u = edge_nodes[idx, 1]
        v = edge_nodes[idx, 2]
        
        dx = x[u, 1] - x[v, 1]
        dy = x[u, 2] - x[v, 2]
        dz = x[u, 3] - x[v, 3]
        
        len = sqrt(dx*dx + dy*dy + dz*dz)
        L[idx] = len
        F[idx] = q[idx] * len
    end
    return nothing
end

"""
    kernel_compute_edge_gradients!(x, lambda, edge_nodes, dq)

Compute adjoint gradients for each edge.
`dq_i = -(x_u - x_v) ⋅ (λ_u - λ_v)`
"""
function kernel_compute_edge_gradients!(x, lambda, edge_nodes, dq)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= size(edge_nodes, 1)
        u = edge_nodes[idx, 1]
        v = edge_nodes[idx, 2]
        
        dx = x[u, 1] - x[v, 1]
        dy = x[u, 2] - x[v, 2]
        dz = x[u, 3] - x[v, 3]
        
        dlx = lambda[u, 1] - lambda[v, 1]
        dly = lambda[u, 2] - lambda[v, 2]
        dlz = lambda[u, 3] - lambda[v, 3]
        
        dq[idx] = -(dx*dlx + dy*dly + dz*dlz)
    end
    return nothing
end

"""
    kernel_update_fixed_nodes!(x_full, fixed_pos, fixed_indices)

Copy fixed node positions from a dense Matrix to the full nodal buffer.
"""
function kernel_update_fixed_nodes!(x_full, fixed_pos, fixed_indices)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(fixed_indices)
        node_idx = Int(fixed_indices[idx])
        x_full[node_idx, 1] = fixed_pos[idx, 1]
        x_full[node_idx, 2] = fixed_pos[idx, 2]
        x_full[node_idx, 3] = fixed_pos[idx, 3]
    end
    return nothing
end

"""
    kernel_scatter_x!(x_full, x_free, free_indices)

Scatter free node positions to the full nodal buffer.
"""
function kernel_scatter_x!(x_full, x_free, free_indices)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(free_indices)
        node_idx = Int(free_indices[idx])
        x_full[node_idx, 1] = x_free[idx, 1]
        x_full[node_idx, 2] = x_free[idx, 2]
        x_full[node_idx, 3] = x_free[idx, 3]
    end
    return nothing
end

"""
    kernel_scatter_lambda!(lambda_full, lambda_free, free_indices)

Scatter free node sensitivities to the full nodal sensitivity buffer.
"""
function kernel_scatter_lambda!(lambda_full, lambda_free, free_indices)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(free_indices)
        node_idx = Int(free_indices[idx])
        lambda_full[node_idx, 1] = lambda_free[idx, 1]
        lambda_full[node_idx, 2] = lambda_free[idx, 2]
        lambda_full[node_idx, 3] = lambda_free[idx, 3]
    end
    return nothing
end

"""
    kernel_accumulate_nodal_sensitivity!(grad_x, dL, dF, q, x, L, edge_nodes, color_group)

Race-free accumulation of nodal sensitivities from edge-based objectives.
"""
function kernel_accumulate_nodal_sensitivity!(grad_x, dL, dF, q, x, L, edge_nodes, color_group)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(color_group)
        e_idx = Int(color_group[idx])
        u = Int(edge_nodes[e_idx, 1])
        v = Int(edge_nodes[e_idx, 2])
        
        inv_L = 1.0 / L[e_idx]
        ux = (x[u, 1] - x[v, 1]) * inv_L
        uy = (x[u, 2] - x[v, 2]) * inv_L
        uz = (x[u, 3] - x[v, 3]) * inv_L
        
        coeff = dL[e_idx] + q[e_idx] * dF[e_idx]
        
        grad_x[u, 1] += coeff * ux
        grad_x[u, 2] += coeff * uy
        grad_x[u, 3] += coeff * uz
        
        grad_x[v, 1] -= coeff * ux
        grad_x[v, 2] -= coeff * uy
        grad_x[v, 3] -= coeff * uz
    end
    return nothing
end

"""
    kernel_admm_z_update!(x, edge_nodes, y, z, L_min, L_max, F_target, q)

Perform ADMM z-update (independent edge projections).
`z` is the consensus edge vector.
"""
function kernel_admm_z_update!(x, edge_nodes, y, z, L_min, L_max, F_target, q)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= size(edge_nodes, 1)
        u = Int(edge_nodes[idx, 1])
        v = Int(edge_nodes[idx, 2])
        
        # current edge vector from x (nodal positions)
        dx = x[u, 1] - x[v, 1]
        dy = x[u, 2] - x[v, 2]
        dz = x[u, 3] - x[v, 3]
        
        # Length constraints
        L = sqrt(dx*dx + dy*dy + dz*dz)
        L_target = L
        if L_min[idx] > 0 || L_max[idx] < 1e10
            L_target = clamp(L, L_min[idx], L_max[idx])
        end
        
        # Force constraints (overrides length if present)
        if F_target[idx] > 0
            L_target = F_target[idx] / max(q[idx], 1e-12)
        end
        
        # Consensus RHS: (x_u - x_v) + y_idx
        rhs_x = dx + y[idx, 1]
        rhs_y = dy + y[idx, 2]
        rhs_z = dz + y[idx, 3]
        rhs_len = sqrt(rhs_x^2 + rhs_y^2 + rhs_z^2)
        
        # Project onto sphere of radius L_target
        scale = L_target / max(rhs_len, 1e-12)
        z[idx, 1] = rhs_x * scale
        z[idx, 2] = rhs_y * scale
        z[idx, 3] = rhs_z * scale
    end
    return nothing
end

"""
    kernel_admm_dual_q_update!(x, edge_nodes, z, y, q, rho)

Update dual variables and force densities.
`y = y + (x_u - x_v) - z`
`q = scale * q` (heuristic)
"""
function kernel_admm_dual_q_update!(x, edge_nodes, z, y, q, rho)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= size(edge_nodes, 1)
        u = Int(edge_nodes[idx, 1])
        v = Int(edge_nodes[idx, 2])
        
        dx = x[u, 1] - x[v, 1]
        dy = x[u, 2] - x[v, 2]
        dz = x[u, 3] - x[v, 3]
        
        # Dual update: y = y + (Δx - z)
        y[idx, 1] += (dx - z[idx, 1])
        y[idx, 2] += (dy - z[idx, 2])
        y[idx, 3] += (dz - z[idx, 3])
        
        # Force density update (Heuristic consensus)
        dist = sqrt(dx*dx + dy*dy + dz*dz)
        target_dist = sqrt(z[idx,1]^2 + z[idx,2]^2 + z[idx,3]^2)
        
        # Update q such that q*dist moves towards target tension
        # This is a relaxed update to maintain stability
        q[idx] = q[idx] * (0.8 + 0.2 * (target_dist / max(dist, 1e-6)))
    end
    return nothing
end

"""
    kernel_compute_reactions!(reactions, x, edge_nodes, q, fixed_node_to_fixed_idx)

Compute reaction forces at fixed nodes.
`reactions`: (N_fixed, 3) output.
`x`: (N_total, 3) positions.
`edge_nodes`: (M, 2) connectivity.
`q`: (M, 1) force densities.
`fixed_node_to_fixed_idx`: (N_total, 1) mapping from global node index to fixed block index (0 if free).
"""
function kernel_compute_reactions!(reactions, x, edge_nodes, q, fixed_node_to_fixed_idx)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= size(edge_nodes, 1)
        u = Int(edge_nodes[idx, 1])
        v = Int(edge_nodes[idx, 2])
        
        q_i = q[idx]
        dx = q_i * (x[u, 1] - x[v, 1])
        dy = q_i * (x[u, 2] - x[v, 2])
        dz = q_i * (x[u, 3] - x[v, 3])
        
        # If u is fixed, reaction at u gets -axial_vector
        idx_u = Int(fixed_node_to_fixed_idx[u])
        if idx_u > 0
            CUDA.@atomic reactions[idx_u, 1] -= dx
            CUDA.@atomic reactions[idx_u, 2] -= dy
            CUDA.@atomic reactions[idx_u, 3] -= dz
        end
        
        # If v is fixed, reaction at v gets +axial_vector (since it's at v)
        # Wait, the incidence matrix C_ij is 1 at u and -1 at v.
        # Edge vector is (x_u - x_v).
        # axial_vector = q_i * (x_u - x_v).
        # R = -C' * axial_vector.
        # R_u = -C_iu * axial_vector = -1 * (q_i * (x_u - x_v)) = -q_i(x_u - x_v).
        # R_v = -C_iv * axial_vector = -(-1) * (q_i * (x_u - x_v)) = +q_i(x_u - x_v).
        
        idx_v = Int(fixed_node_to_fixed_idx[v])
        if idx_v > 0
            CUDA.@atomic reactions[idx_v, 1] += dx
            CUDA.@atomic reactions[idx_v, 2] += dy
            CUDA.@atomic reactions[idx_v, 3] += dz
        end
    end
    return nothing
end

"""
    kernel_softplus_penalty!(val, threshold, dval, weight, sharpness)

Compute C^2 softplus penalty gradients with respect to values (lengths or forces).
"""
function kernel_softplus_penalty!(val, threshold, dval, weight, sharpness)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(val)
        v = val[idx]
        t = threshold[idx]
        alpha = sharpness
        
        z = alpha * (v - t)
        if z > 50.0 
            dval[idx] = weight
        elseif z < -50.0
            dval[idx] = 0.0
        else
            ez = exp(z)
            dval[idx] = weight * ez / (1.0 + ez)
        end
    end
    return nothing
end

"""
    kernel_target_xyz_sensitivity!(xyz, target, indices, grad_x, weight)

Compute sensitivities for TargetXYZObjective directly into grad_x.
"""
function kernel_target_xyz_sensitivity!(xyz, target, indices, grad_x, weight)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(indices)
        n_idx = Int(indices[idx])
        
        grad_x[n_idx, 1] += 2.0 * weight * (xyz[n_idx, 1] - target[idx, 1])
        grad_x[n_idx, 2] += 2.0 * weight * (xyz[n_idx, 2] - target[idx, 2])
        grad_x[n_idx, 3] += 2.0 * weight * (xyz[n_idx, 3] - target[idx, 3])
    end
    return nothing
end

