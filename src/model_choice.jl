function calculate_energy_density(env_arr, loc, loc_d, Pattern_arr; Space_type = ℝ, Ham_parameters = nothing, model = :Heisenberg_square, lattice = :square, u_paras = false)
   
    if lattice == :square
        energy_density = exp_val_energy_generic_square_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    elseif lattice == :kagome
        energy_density = exp_val_energy_generic_kagome_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    elseif lattice == :honeycomb
        energy_density = exp_val_energy_generic_honeycomb_lattice(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    elseif lattice == :triangular
        energy_density = exp_val_energy_generic_triangular_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type, u_paras = u_paras)
    elseif lattice == :dice
        energy_density = exp_val_energy_generic_dice_lattice(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    end

    return energy_density
end

function calculate_observables(env_arr, loc, loc_d, Pattern_arr; Space_type = Space_type, Ham_parameters = nothing, model = :Heisenberg_square, lattice = :lattice)

    if lattice == :square
        observables = exp_val_observables_generic_square_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    elseif lattice == :kagome
        observables = exp_val_observables_generic_kagome_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    elseif lattice == :honeycomb
        observables = exp_val_observables_generic_honeycomb_lattice(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    elseif lattice == :triangular
        observables = exp_val_observables_generic_triangular_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    elseif lattice == :dice
        observables = exp_val_observables_generic_dice_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = model, Space_type = Space_type)
    end

    return observables
    
end
