# throw(ErrorException("Oppps! No methods defined in src/Factory.jl. What should you do here?"))

# src/Factory.jl

# -- PUBLIC METHODS BELOW HERE ---------------------------------------------------------------------------------------- #
"""
    build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple) -> MyClassicalHopfieldNetworkModel

Factory method for building a Hopfield network model using Hebbian learning.

### Arguments
- `modeltype::Type{MyClassicalHopfieldNetworkModel}`: the type of the model to be built.
- `data::NamedTuple`: a named tuple containing the data for the model.

The named tuple should contain the following fields:
- `memories`: a matrix of memories where each column is a memory pattern (N×K matrix, values ∈ {-1, 1}).

### Returns
- `model::MyClassicalHopfieldNetworkModel`: the built Hopfield network model with the following fields populated:
    - `W`: the weight matrix (symmetric, zero diagonal).
    - `b`: the bias vector (all zeros for classical Hopfield).
    - `energy`: a dictionary of energies for each memory pattern.
"""
function build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)::MyClassicalHopfieldNetworkModel

    # initialize -
    linearimagecollection = data.memories  # Each column is a memory pattern
    number_of_rows, number_of_cols = size(linearimagecollection)
    
    # Create model instance using the empty constructor
    model = modeltype()
    
    # Initialize weight matrix and bias vector
    W = zeros(Float64, number_of_rows, number_of_rows)
    b = zeros(Float64, number_of_rows)  # zero bias for classical Hopfield

    # Compute the weight matrix using Hebbian learning rule
    # W = (1/K) * Σᵢ sᵢ ⊗ sᵢᵀ
    for j ∈ 1:number_of_cols
        Y = ⊗(linearimagecollection[:, j], linearimagecollection[:, j])  # outer product
        W += Y  # accumulate outer products
    end
    
    # Apply Hebbian scaling and ensure no self-coupling
    for i ∈ 1:number_of_rows
        W[i, i] = 0.0  # no self-coupling in classical Hopfield network
    end
    WN = (1 / number_of_cols) * W  # Hebbian scaling by number of memories
    
    # Compute the energy dictionary for each stored memory
    energy = Dict{Int64, Float32}()
    for i ∈ 1:number_of_cols
        energy[i] = _energy(linearimagecollection[:, i], WN, b)
    end

    # Populate the model with computed values
    model.W = WN
    model.b = b
    model.energy = energy

    # return -
    return model
end
# --- PUBLIC METHODS ABOVE HERE --------------------------------------------------------------------------------------- #