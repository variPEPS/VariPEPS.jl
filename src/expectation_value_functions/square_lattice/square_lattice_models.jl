function use_Heisenberg_square_lattice(Ham_parameters, loc, pattern_arr; Space_type = ℂ)

    if Space_type != :U1 && Space_type != :Z2
        #display("ola mista")
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

    elseif Space_type == :U1

        #display("hello")
        h = Ham_parameters.h
        J = Ham_parameters.J
        dir = Ham_parameters.dir
    
        int_term = Heisenberg_interaction_operators_U1(; Space_type = :U1)
    
        vertical_term = J * int_term
    
        horizontal_term = J * int_term
    
        ham_term_arr = []
    
        for i in minimum(pattern_arr):maximum(pattern_arr)
            push!(ham_term_arr, (hor = horizontal_term, vert = vertical_term))
        end

        return ham_term_arr

    elseif Space_type == :Z2

        Sx = 1/2 * TensorMap([0 +1 ; +1 0], ComplexSpace(2), ComplexSpace(2));
        Sy = 1/2 * TensorMap([0 -1im ; +1im 0], ComplexSpace(2), ComplexSpace(2));
        Sz = 1/2 * TensorMap([+1 0 ; 0 -1], ComplexSpace(2), ComplexSpace(2));
        nonSymH = Sx ⊗ Sx + Sy ⊗ Sy + Sz ⊗ Sz

        P = Z2Space(0 => 1, 1 => 1)
        hamZ2 = TensorMap(convert(Array, nonSymH), P ⊗ P, P ⊗ P)

        ham_term_arr = []
        for i in minimum(pattern_arr):maximum(pattern_arr)
            push!(ham_term_arr, (hor = hamZ2, vert = hamZ2))
        end

        return ham_term_arr
    end
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









