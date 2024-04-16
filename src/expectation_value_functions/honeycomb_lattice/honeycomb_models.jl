function use_Kitaev_Γ_model(Ham_parameters, loc, pattern_arr; Space_type = ℂ)

    #Kitaev_term_x, Kitaev_term_y, Kitaev_term_z , σ_x, σ_y, σ_z, Ham_term_Γ_x, Ham_term_Γ_y, Ham_term_Γ_z = Zygote.@ignore create_spin_operators_kitaev_Γ(; Space_type = Space_type)
    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = Space_type)

    @tensor Ham_term_Γ_x[(i, j);(k, l)] :=  σ_y[i,k] * σ_z[j, l] + σ_z[i,k] * σ_y[j, l]
    @tensor Ham_term_Γ_y[(i, j);(k, l)] :=  σ_x[i,k] * σ_z[j, l] + σ_z[i,k] * σ_x[j, l]
    @tensor Ham_term_Γ_z[(i, j);(k, l)] :=  σ_x[i,k] * σ_y[j, l] + σ_y[i,k] * σ_x[j, l]
    
    @tensor Kitaev_term_x[(i, j);(k, l)] :=  σ_x[i,k] * σ_x[j, l]
    @tensor Kitaev_term_y[(i, j);(k, l)] :=  σ_y[i,k] * σ_y[j, l]
    @tensor Kitaev_term_z[(i, j);(k, l)] :=  σ_z[i,k] * σ_z[j, l]
    
    h = Ham_parameters.h
    Φ = Ham_parameters.Φ

    if Φ == 1
        pseudo_loc_term = Kitaev_term_x
    
        horizontal_term = Kitaev_term_y
    
        vertical_term = Kitaev_term_z
        
    else
        
        pseudo_loc_term = -cos(Φ * π) * Kitaev_term_x + sin(Φ * π) * Ham_term_Γ_x
    
        horizontal_term = -cos(Φ * π) * Kitaev_term_y + sin(Φ * π) * Ham_term_Γ_y
    
        vertical_term = -cos(Φ * π) * Kitaev_term_z + sin(Φ * π) * Ham_term_Γ_z
    
    end

    if h != 0
        local_terms_A = (-2*h/sqrt(3)) * (σ_x + σ_y + σ_z)
        local_terms_B = (-2*h/sqrt(3)) * (σ_x + σ_y + σ_z)

        @tensor loc_term_A_2legs[(i, j);(k, l)] := local_terms_A[i,k] * id[j, l]
        @tensor loc_term_B_2legs[(i, j);(k, l)] := id[i,k] * local_terms_B[j, l]
        loc_term = pseudo_loc_term + loc_term_A_2legs + loc_term_A_2legs
    else
        loc_term = pseudo_loc_term
    end

    ham_term_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(ham_term_arr, (loc = loc_term, hor = horizontal_term, vert = vertical_term))
    end


    return ham_term_arr
end


function use_Heisenberg_honeycomb(Ham_parameters, loc, pattern_arr; Space_type = ℂ)

    #Kitaev_term_x, Kitaev_term_y, Kitaev_term_z , σ_x, σ_y, σ_z, Ham_term_Γ_x, Ham_term_Γ_y, Ham_term_Γ_z = Zygote.@ignore create_spin_operators_kitaev_Γ(; Space_type = Space_type)
    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = Space_type)

    @tensor Heisenberg_term[(i, j);(k, l)] :=  (σ_p[i,k] * σ_m[j, l] + σ_m[i,k] * σ_p[j,l])/2 + (σ_z[i,k] * σ_z[j,l])
    
    J = Ham_parameters.J

    loc_term = J * Heisenberg_term
    
    horizontal_term = J * Heisenberg_term
    
    vertical_term = J * Heisenberg_term
        
    #local_terms_A = (-2*h/sqrt(3)) * (σ_x + σ_y + σ_z)
    #local_terms_B = (-2*h/sqrt(3)) * (σ_x + σ_y + σ_z)

    ham_term_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(ham_term_arr, (loc = loc_term, hor = horizontal_term, vert = vertical_term))
    end

    return ham_term_arr
end