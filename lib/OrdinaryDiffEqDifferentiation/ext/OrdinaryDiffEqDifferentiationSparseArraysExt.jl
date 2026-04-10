module OrdinaryDiffEqDifferentiationSparseArraysExt

using OrdinaryDiffEqDifferentiation
using LinearAlgebra
import SparseArrays
import SparseArrays: nonzeros, spzeros, SparseMatrixCSC, AbstractSparseMatrix

# Override the sparse checking functions
OrdinaryDiffEqDifferentiation.is_sparse(::AbstractSparseMatrix) = true
OrdinaryDiffEqDifferentiation.is_sparse_csc(::SparseMatrixCSC) = true

# Override the sparse array manipulation functions
OrdinaryDiffEqDifferentiation.nonzeros(A::AbstractSparseMatrix) = nonzeros(A)
OrdinaryDiffEqDifferentiation.spzeros(T::Type, m::Integer, n::Integer) = spzeros(T, m, n)

# Helper functions for accessing sparse matrix internals
OrdinaryDiffEqDifferentiation.get_nzval(A::SparseMatrixCSC) = A.nzval
OrdinaryDiffEqDifferentiation.set_all_nzval!(A::SparseMatrixCSC, val) = (A.nzval .= val; A)

# Fast path for sparse W and J (assumed equal patterns) and M = I
# Avoids [i, i] lookups, which triggers searching
function OrdinaryDiffEqDifferentiation.jacobian2W!(W::SparseMatrixCSC, M::UniformScaling, γdt::Number, J::SparseMatrixCSC)::Nothing
    size(W) == size(J) || _throwWJerror(W, J)
    λ_γdt = M.λ * inv(γdt)
    @inbounds for j in 1:size(W, 2)
        @inbounds @simd for k in W.colptr[j]:W.colptr[j+1]-1
            i = W.rowval[k]
            W.nzval[k] = i == j ? J.nzval[k] - λ_γdt : J.nzval[k]
        end
    end
    return nothing
end

end
