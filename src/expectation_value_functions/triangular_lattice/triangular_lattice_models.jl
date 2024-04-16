function use_Heisenberg_triangular_lattice(Ham_parameters, loc, pattern_arr; Space_type = ℂ)

    h = Ham_parameters.h
    Jxy = Ham_parameters.Jxy
    Jdiag = Ham_parameters.Jdiag
    dir = Ham_parameters.dir

    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = Space_type)

    #in this way the tensor is real valued.
    @tensor int_term[(i, j);(k, l)] :=  (σ_p[i,k] * σ_m[j, l] + σ_m[i,k] * σ_p[j,l])/2 + (σ_z[i,k] * σ_z[j,l])

    #add a local magnetic field if wanted
    #loc_term = h * (dir[1] *  σ_x + dir[2] * σ_y + dir[3] * σ_z)

    vertical_term = Jxy * int_term

    horizontal_term = Jxy * int_term

    diag_term = Jdiag * int_term

    ham_term_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(ham_term_arr, (hor = horizontal_term, vert = vertical_term, diag = diag_term))
    end

    return ham_term_arr
end