# Set up the environment to run this example. Make sure you're within the folder that this
# file lives in.
using Pkg
Pkg.activate(@__DIR__)
# !!! Strongly suggests to run "git clone git@github.com:quantumlang/QuantumLangBase.jl.git", and then
# use `Pkg.add(path="path to QuantumLangBase dir")`
Pkg.add(url="https://github.com/quantumlang/QuantumLangBase.jl")
Pkg.instantiate()

using QuantumLangBase
using Functors
import Functors: functor
using Zygote
using Random
using Test

rng = MersenneTwister(123456)
V = Float64

# In quantum mechanics, we often know the function that generate the operator's matrix
# element, e.g., ⟨k|V|k′⟩=V̂(k-k′). Here, we assume this function to be V̂(k-k′)=cos((k-k′)a)
struct TestFunc
    k::Parameter     # Parameter is just a wrapper for variational variable, it's value can be extracted by calling `value`
    a::Constant      # Constant is a wrapper for non-differentable variable, it's value can also be extracted by calling `value`
end

# Quantum mechanics denote a quantum state uniquely with it's quantum number, e.g., (major, angular, magnetic)
# quantum number is used for hydrogen atom. `MultiIndex` is used to mimic this behavior, it's designed to be
# an easy way to index states in Hilbert space.
function (tf::TestFunc)(mi::MultiIndex, mj::MultiIndex)
    i, j = mi[1], mj[1]
    return cos((value(tf.k)[i] - value(tf.k)[j])*value(tf.a))
end

# `functor` is here to extract all the field values in `TestFunc`, we will map data type of `Parameter` in `TestFunc` to dual number
# when evaluate gradient
function functor(tf::TestFunc)
    return (k=tf.k, a=tf.a), function test_f_back(v)
        TestFunc(v...)
    end
end

# To summarize, given a `MatrixOperator` and pass in two `MultiIndex`, you will obtain a matrix element


s1 = MultiIndex((1,), (1,))
s2 = MultiIndex((10,), (1,))
const ax = s1:s2

kp = Parameter(rand(rng, V, 10))
ap = Constant(V(1.0))
test_f = TestFunc(kp, ap)
test_Op = MatrixOperator(test_f, (ax, ax))
# `Array{V}` is overloaded for usage in QuantumLangBase, you need to
# act it on a MatrixOperator/TensorOperator instance to get an array,
# see here for detail:
# https://github.com/quantumlang/QuantumLangBase.jl/blob/460c7761dc8b2e6fc59d9fe8c2a2b860c0a9e97a/src/operator.jl#L44-L51
Array{V}(test_Op)

# loss function
const x, y = rand(rng, V, 10), rand(rng, V, 10)

function loss(op::MatrixOperator)
    mtrx = Array{V}(op)
    return x'*mtrx*y
end

@show loss(test_Op)
@show gradient(loss, test_Op)

# We can explicitly write it out in loss
function lossexpand1(op::MatrixOperator)
    n, m = size(op)
    A = Matrix{V}(undef, n, m)            # allocate a matrix to store matrix elements
    for j in 1:m, i in 1:n
        # `getindex` is overloaded for `MatrixOperator`, it's definition can be found here 
        # https://github.com/quantumlang/QuantumLangBase.jl/blob/460c7761dc8b2e6fc59d9fe8c2a2b860c0a9e97a/src/operator.jl#L37-L42
        A[i, j] = getindex(op, i, j)
    end
    return x'*A*y
end

@test lossexpand1(test_Op) ≈ loss(test_Op)


# if we fully expand all the functions
const a = one(V)
const N, M = 10, 10

function lossexpand2(kp::Vector{T}) where {T<:Real}
    A = Matrix{V}(undef, N, M)
    for j in 1:M, i in 1:N
        mi, mj = ax[i], ax[j]
        A[i, j] = cos((kp[mi[1]]-kp[mj[1]])*a)
    end
    return x'*A*y
end

@test lossexpand2(value(kp)) ≈ loss(test_Op)

