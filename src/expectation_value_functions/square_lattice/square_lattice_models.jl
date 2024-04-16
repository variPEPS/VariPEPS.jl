function use_Heisenberg_square_lattice(Ham_parameters, loc, pattern_arr; Space_type = ℂ)

    h = Ham_parameters.h
    J = Ham_parameters.J
    dir = Ham_parameters.dir

    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = Space_type)

    #in this way the tensor is real valued.
    @tensor int_term[(i, j);(k, l)] :=  (σ_p[i,k] * σ_m[j, l] + σ_m[i,k] * σ_p[j,l])/2 + (σ_z[i,k] * σ_z[j,l])

    #add a local magnetic field if wanted
    loc_term = h * (dir[1] *  σ_x + dir[2] * σ_y + dir[3] * σ_z)

    vertical_term = J * int_term

    horizontal_term = J * int_term

    ham_term_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(ham_term_arr, (loc = loc_term, hor = horizontal_term, vert = vertical_term))
    end

    return ham_term_arr
end

function use_J1J2model_square_lattice(Ham_parameters, loc, pattern_arr; Space_type = ℂ)

    h = Ham_parameters.h
    J1 = Ham_parameters.J1
    J2 = Ham_parameters.J2
    dir = Ham_parameters.dir

    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = Space_type)

    #in this way the tensor is real valued.
    @tensor int_term[(i, j);(k, l)] :=  (σ_p[i,k] * σ_m[j, l] + σ_m[i,k] * σ_p[j,l])/2 + (σ_z[i,k] * σ_z[j,l])

    #add a local magnetic field if wanted
    loc_term = h * (dir[1] *  σ_x + dir[2] * σ_y + dir[3] * σ_z)

    vertical_term = J1 * int_term

    horizontal_term = J1 * int_term

    diag_ur_term = J2 * int_term

    diag_dr_term = J2 * int_term

    ham_term_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(ham_term_arr, (loc = loc_term, hor = horizontal_term, vert = vertical_term, diag_ur = diag_ur_term, diag_dr = diag_dr_term))
    end

    return ham_term_arr
end









