# throw(ErrorException("Oppps! No methods defined in src/Compute.jl. What should you do here?"))

# src/Compute.jl

# Load required packages at the top level
using DataStructures

"""
    _energy(s::Array{<:Number,1}, W::Array{<:Number,2}, b::Array{<:Number,1}) -> Float32

Compute the energy of state s given weights W and bias b.
Energy formula: E(s) = -½ Σᵢⱼ wᵢⱼsᵢsⱼ - Σᵢ bᵢsᵢ
"""
function _energy(s::Array{<:Number,1}, W::Array{<:Number,2}, b::Array{<:Number,1})::Float32
    
    # initialize -
    tmp_energy_state = 0.0
    number_of_states = length(s)

    # main loop -
    tmp = transpose(b) * s  # bias term
    for i ∈ 1:number_of_states
        for j ∈ 1:number_of_states
            tmp_energy_state += W[i,j] * s[i] * s[j]
        end
    end
    energy_state = -(1/2) * tmp_energy_state - tmp

    # return -
    return Float32(energy_state)
end

"""
    ⊗(a::Array{T,1}, b::Array{T,1}) -> Array{T,2} where T <: Number

Compute the outer product of two vectors a and b and returns a matrix.
"""
function ⊗(a::Array{T,1}, b::Array{T,1})::Array{T,2} where T <: Number

    # initialize -
    m = length(a)
    n = length(b)
    Y = zeros(T, m, n)

    # main loop 
    for i ∈ 1:m
        for j ∈ 1:n
            Y[i,j] = a[i] * b[j]
        end
    end

    # return 
    return Y
end

"""
    hamming(a::Array{T,1}, b::Array{T,1}) -> Int where T <: Number

Compute Hamming distance between two binary vectors.
"""
function hamming(a::Array{T,1}, b::Array{T,1})::Int where T <: Number
    return sum(a .!= b)
end

"""
    recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, 
            true_image_energy::Float32; maxiterations::Int64=1000, 
            patience::Union{Int,Nothing}=nothing,
            miniterations_before_convergence::Union{Int,Nothing}=nothing)
    -> Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}}

Run asynchronous Hopfield updates starting from sₒ until convergence.

### Arguments
- `model::MyClassicalHopfieldNetworkModel`: a Hopfield network model
- `sₒ::Array{Int32,1}`: initial corrupted state (±1 values)
- `true_image_energy::Float32`: energy of the target memory pattern
- `maxiterations::Int64`: maximum number of update iterations (default: 1000)
- `patience::Union{Int,Nothing}`: consecutive identical states for convergence (default: auto-scaled)
- `miniterations_before_convergence::Union{Int,Nothing}`: minimum iterations before checking convergence

### Returns
- `frames::Dict{Int64, Array{Int32,1}}`: state at each iteration
- `energydictionary::Dict{Int64, Float32}`: energy at each iteration
"""
function recover(model::MyClassicalHopfieldNetworkModel, 
                 sₒ::Array{Int32,1}, 
                 true_image_energy::Float32;
                 maxiterations::Int64=1000, 
                 patience::Union{Int,Nothing}=nothing,
                 miniterations_before_convergence::Union{Int,Nothing}=nothing)::Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}}

    # initialize -
    W = model.W  # get the weights
    b = model.b  # get the biases
    number_of_pixels = length(sₒ)  # number of pixels
    
    # Set patience value with scaling
    patience_val = isnothing(patience) ? max(5, Int(round(0.01 * number_of_pixels))) : patience
    
    # Set minimum iterations before convergence check
    min_iterations = isnothing(miniterations_before_convergence) ? patience_val : miniterations_before_convergence
    min_iterations = max(min_iterations, patience_val)  # floor before declaring convergence
    
    # Circular buffer to check for state stability
    S = CircularBuffer{Array{Int32,1}}(patience_val)
    
    # Storage dictionaries
    frames = Dict{Int64, Array{Int32,1}}()
    energydictionary = Dict{Int64, Float32}()
    has_converged = false
    
    # Store initial state
    frames[0] = copy(sₒ)
    energydictionary[0] = _energy(sₒ, W, b)
    s = copy(sₒ)  # current state
    iteration_counter = 1
    
    # Main iteration loop
    while has_converged == false
        
        # Asynchronous update: select a random neuron
        j = rand(1:number_of_pixels)
        w = W[j, :]  # weights for neuron j
        
        # Compute activation
        h = dot(w, s) - b[j]
        
        # Update state with tie-breaking for h == 0
        if h == 0
            s[j] = rand() < 0.5 ? Int32(-1) : Int32(1)  # random tie-break
        else
            s[j] = h > 0 ? Int32(1) : Int32(-1)
        end
        
        # Store current state and energy
        state_snapshot = copy(s)
        frames[iteration_counter] = state_snapshot
        energydictionary[iteration_counter] = _energy(s, W, b)
        
        # Check for convergence: state stability
        push!(S, state_snapshot)
        if (length(S) == patience_val) && (iteration_counter >= min_iterations)
            all_equal = true
            first_state = S[1]
            for state ∈ S
                if hamming(first_state, state) != 0
                    all_equal = false
                    break
                end
            end
            if all_equal == true
                has_converged = true
            end
        end
        
        # Check for convergence: energy criterion
        current_energy = energydictionary[iteration_counter]
        if current_energy ≤ true_image_energy
            has_converged = true
        end
        
        # Update counter and check max iterations
        iteration_counter += 1
        if (iteration_counter > maxiterations) && (has_converged == false)
            has_converged = true
            @warn "Maximum iterations reached without convergence."
        end
    end
            
    # return 
    return frames, energydictionary
end

"""
    decode(simulationstate::Array{T,1}; number_of_rows::Int64=28, 
           number_of_cols::Int64=28) -> Array{T,2} where T <: Number

Decode a linear state vector back into a 2D image array.
Converts -1 → 0 (black) and 1 → 1 (white).
"""
function decode(simulationstate::Array{T,1}; 
                number_of_rows::Int64=28, 
                number_of_cols::Int64=28)::Array{T,2} where T <: Number
    
    # initialize -
    reconstructed_image = Array{Int32,2}(undef, number_of_rows, number_of_cols)
    linearindex = 1
    
    for row ∈ 1:number_of_rows
        for col ∈ 1:number_of_cols
            s = simulationstate[linearindex]
            if s == -1
                reconstructed_image[row, col] = 0
            else
                reconstructed_image[row, col] = 1
            end
            linearindex += 1
        end
    end
    
    # return 
    return reconstructed_image
end