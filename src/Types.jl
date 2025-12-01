# throw(ErrorException("Oppps! No methods defined in src/Types.jl. What should you do here?"))


abstract type AbstractHopfieldNetworkModel end

"""
    MyClassicalHopfieldNetworkModel <: AbstractHopfieldNetworkModel

A mutable struct representing a classical Hopfield network model.

### Fields
- `W::Array{<:Number, 2}`: weight matrix (symmetric, zero diagonal).
- `b::Array{<:Number, 1}`: bias vector (typically all zeros).
- `energy::Dict{Int64, Float32}`: energy of each stored memory pattern.
"""
mutable struct MyClassicalHopfieldNetworkModel <: AbstractHopfieldNetworkModel

    # data -
    W::Array{<:Number, 2}          # weight matrix
    b::Array{<:Number, 1}          # bias vector
    energy::Dict{Int64, Float32}   # energy of the states

    # empty constructor -
    MyClassicalHopfieldNetworkModel() = new()
end