module VariPEPS

using LinearAlgebra
using TensorKit
using KrylovKit
using Random
using Zygote
using Zygote: Buffer, @ignore
using ChainRulesCore
using Statistics

export ctmrg, initialize_PEPS, converge_environment, energy_and_gradient, convert_input, create_spin_operators_SU, ℂ, ℝ

#all sorts of custom functions and if needed their pullbacks for AD
include("special_functions.jl")
include("pullbacks.jl")

#the CTMRG setup
include("CTMRG_moves.jl")
include("CTMRG_step.jl")
include("CTMRG_procedure.jl")
include("model_choice.jl")

#calculating the custom gradient for at the fixed point of the CTMRG routine
include("gradient_at_fp.jl")

#expectation value functions for various models and lattices
include("expectation_value_functions/model_operators.jl")

#square lattice_stuff
include("expectation_value_functions/square_lattice/square_lattice_models.jl")
include("expectation_value_functions/square_lattice/observables_square_lattice.jl")
include("expectation_value_functions/square_lattice/exp_val_square_lattice.jl")

#honeycomb_lattice_stuff
include("expectation_value_functions/honeycomb_lattice/exp_val_honeycomb.jl")
include("expectation_value_functions/honeycomb_lattice/honeycomb_models.jl")
include("expectation_value_functions/honeycomb_lattice/honeycomb_observables.jl")

#kagome_lattice_stuff
include("expectation_value_functions/kagome_lattice/kagome_lattice_models.jl")
include("expectation_value_functions/kagome_lattice/exp_val_kagome_lattice.jl")
include("expectation_value_functions/kagome_lattice/kagome_lattice_observables.jl")

#triangular lattice stuff
include("expectation_value_functions/triangular_lattice/exp_val_triangular_lattice.jl")
include("expectation_value_functions/triangular_lattice/triangular_lattice_models.jl")
include("expectation_value_functions/triangular_lattice/observables_triangular_lattice.jl")

#dice lattice stuff
include("expectation_value_functions/dice_lattice/dice_lattice_models.jl")
include("expectation_value_functions/dice_lattice/dice_lattice_observables.jl")
include("expectation_value_functions/dice_lattice/exp_val_dice.jl")

#dictionary for predifined models
include("expectation_value_functions/model_dictionary.jl")
end