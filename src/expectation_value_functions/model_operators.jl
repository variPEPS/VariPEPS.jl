function create_spin_onehalf_operators(; Space_type = ℂ)
    Dim_loc = 2
    space_loc = Space_type^Dim_loc
    
    σ_p_mat =  [0 1 ; 0 0]
    σ_m_mat =  [0 0 ; 1 0]
    σ_z_mat =  0.5 * [1 0 ; 0 -1]
    σ_y_mat =  0.5* [0 -1im ; 1im 0]
    σ_x_mat =  0.5* [0 1 ; 1 0]
    
    σ_p = TensorMap(σ_p_mat, space_loc ← space_loc)
    σ_m = TensorMap(σ_m_mat, space_loc ← space_loc)
    σ_z = TensorMap(σ_z_mat, space_loc ← space_loc)
    σ_y = TensorMap(σ_y_mat, space_loc ← space_loc)
    σ_x = TensorMap(σ_x_mat, space_loc ← space_loc)
    
    #@tensor Ham_term_Γ_x[(i, j);(k, l)] :=  σ_y[i,k] * σ_z[j, l] + σ_z[i,k] * σ_y[j, l]
    #@tensor Ham_term_Γ_y[(i, j);(k, l)] :=  σ_x[i,k] * σ_z[j, l] + σ_z[i,k] * σ_x[j, l]
    #@tensor Ham_term_Γ_z[(i, j);(k, l)] :=  σ_x[i,k] * σ_y[j, l] + σ_y[i,k] * σ_x[j, l]
    
    #@tensor Ham_term_x[(i, j);(k, l)] :=  σ_x[i,k] * σ_x[j, l]
    #@tensor Ham_term_y[(i, j);(k, l)] :=  σ_y[i,k] * σ_y[j, l]
    #@tensor Ham_term_z[(i, j);(k, l)] :=  σ_z[i,k] * σ_z[j, l]


    #@tensor Ham_termy_[(i, j);(k, l)] :=  (σ_p[i,k] * σ_m[j, l] + σ_m[i,k] * σ_p[j,l])/2 + (σ_z[i,k] * σ_z[j,l])
    
    #Ham_term_x = real(Ham_term_x)
    #Ham_term_y = real(Ham_term_y)
    #Ham_term_z = real(Ham_term_z)

    return σ_x, σ_y, σ_z, σ_p, σ_m
end