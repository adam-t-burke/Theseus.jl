function getDiag(sparseDiag::SparseMatrixCSC{T}) where T
    return diag(sparseDiag)
end