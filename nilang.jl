# from: https://github.com/quantumlang/NiLangTest/blob/master/example.jl
using Test
using NiLang
using QuantumLangBase
using NiLang.AD: GVar
NiLang.AD.grad(x::MultiIndex) = nothing

N, M = 100, 100
A = Matrix{Float64}(undef, N, M)
params = randn(Float64, 100)
a = one(Float64)

s1 = MultiIndex((1,), (1,))
s2 = MultiIndex((100,), (1,))
ax = s1:s2
x = randn(Float64, 100)
y = randn(Float64, 100)

@i function lossexpand3(loss, A, kp::Vector{T}, a, x, y, ax) where {T<:Real}
    @invcheckoff @inbounds for j in 1:size(A, 2)
        mj ← ax[j][1]
        for i in 1:size(A, 1)
            mi ← ax[i][1]
            @routine begin
                @zeros T dθ dθa
                dθ += kp[mi]
                dθ -= kp[mj]
                dθa += dθ * a
            end
            A[i, j] += cos(dθa)
            ~@routine
        end
    end

    # compute inner product
    @invcheckoff @inbounds for j in 1:size(A, 2)
        for i in 1:size(A, 1)
            @routine begin
                @zeros T anc
                anc += x[i] * A[i, j]
            end
            loss += anc * y[j]
            ~@routine
        end
    end
end

lossexpand3(0.0, zero(A), params, a, x, y, ax)[1]

using BenchmarkTools
@benchmark NiLang.AD.gradient(Val(1), $lossexpand3, $(0.0, zero(A), params, a, x, y, ax))[3]

