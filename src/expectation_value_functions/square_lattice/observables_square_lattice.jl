function observables_Heisenberg_model_square(ham_parameters, loc, pattern_arr; Space_type = ℂ)

    #here we build the creation an annihilation operators for the truncated local Hilbert spoace of the Bosons
    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = Space_type);

    #=
    here we create an array of the local terms that we want to evaluate for every site in the unit cell.
    =#

    local_obs = [σ_x, σ_y, σ_z]
                         
    local_obs_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(local_obs_arr, (loc = local_obs,))
    end

    return local_obs_arr
end
