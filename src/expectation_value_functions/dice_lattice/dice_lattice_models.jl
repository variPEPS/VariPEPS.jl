function use_Kitaev_Γ_model_dice(ham_parameters, loc, pattern_arr; Space_type = ℂ)

    dim_square = dim(domain(loc[1,1])[3]) #local dimension of the square lattice tensor
    dim_dice = Int(cbrt(dim_square)) #local dimension of the d.o.f. on the dice lattice
    
    split = isomorphism(Space_type^dim_dice ⊗ Space_type^dim_dice ⊗ Space_type^dim_dice, Space_type^dim_square)
    fuse = isomorphism(Space_type^dim_square, Space_type^dim_dice ⊗ Space_type^dim_dice ⊗ Space_type^dim_dice)

    #here we build the creation an annihilation operators for the truncated local Hilbert spoace of the Bosons
    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = ℂ)
   
    h = ham_parameters.h
    Φ = ham_parameters.Φ

    @tensor Ham_term_Γ_x[(i, j);(k, l)] :=  σ_y[i,k] * σ_z[j, l] + σ_z[i,k] * σ_y[j, l]
    @tensor Ham_term_Γ_y[(i, j);(k, l)] :=  σ_x[i,k] * σ_z[j, l] + σ_z[i,k] * σ_x[j, l]
    @tensor Ham_term_Γ_z[(i, j);(k, l)] :=  σ_x[i,k] * σ_y[j, l] + σ_y[i,k] * σ_x[j, l]
    
    @tensor Ising_term_x[(i, j);(k, l)] :=  σ_x[i,k] * σ_x[j, l]
    @tensor Ising_term_y[(i, j);(k, l)] :=  σ_y[i,k] * σ_y[j, l]
    @tensor Ising_term_z[(i, j);(k, l)] :=  σ_z[i,k] * σ_z[j, l]

    local_term = (-2*h/sqrt(3)) * (σ_x + σ_y + σ_z)

    if Φ == 1
        pseudo_loc_term = Ising_term_x
    
        horizontal_term = Ising_term_y
    
        vertical_term = Ising_term_z
        
    else
        
        pseudo_loc_term = -cos(Φ * π) * Ising_term_x + sin(Φ * π) * Ham_term_Γ_x
    
        horizontal_term = -cos(Φ * π) * Ising_term_y + sin(Φ * π) * Ham_term_Γ_y
    
        vertical_term = -cos(Φ * π) * Ising_term_z + sin(Φ * π) * Ham_term_Γ_z
    
    end
    
    local_term_dice = local_term

    #=
    now we create the local term of the model on the coarse grained level. 
    The local term consists of as sum of:
    See Notes for definition of the unit cell.
    1. local terms for the three sites within the UC.
    2. A hopping term from the first to the second site in the unit cell.
    3. A hopping term from the second to the third site in the unit cell.
    =#

    @tensor local_term_1a[(out,);(in,)] := fuse[out, out1, out2, out3] * local_term_dice[out1,in1] * split[in1, out2, out3, in]
    @tensor local_term_1b[(out,);(in,)] := fuse[out, out1, out2, out3] * local_term_dice[out2,in2] * split[out1, in2, out3, in]
    @tensor local_term_1c[(out,);(in,)] := fuse[out, out1, out2, out3] * local_term_dice[out3,in3] * split[out1, out2, in3, in]

    @tensor local_term_2[(out,);(in,)] := fuse[out, out1, out2, out3] * pseudo_loc_term[out1,out2,in1,in2] * split[in1, in2, out3, in]
    @tensor local_term_3[(out,);(in,)] := fuse[out, out1, out2, out3] * pseudo_loc_term[out2,out3,in2,in3] * split[out1, in2, in3, in]

    local_term_coarse = local_term_1a + local_term_1b + local_term_1c + local_term_2 + local_term_3
    
    #=
    now we create the horizontal terms on the coarse grained level.
    the horizontal term consists of:
    1. A hopping term from site 2 of the UC to site 1 of the UC to the right.
    2. A hopping term from site 3 of the UC to site 2 of the UC to the right.
    =#
    
    #these contractions might need to be optimized!
    @tensor begin horizontal_term_1[(out_uc1, out_uc2);(in_uc1,in_uc2)] := fuse[out_uc1, out1_uc1, out2_uc1, out3_uc1] * fuse[out_uc2, out1_uc2, out2_uc2, out3_uc2] * 
                                                                horizontal_term[out2_uc1, out1_uc2, in2_uc1, in1_uc2] *
                                                                split[out1_uc1, in2_uc1, out3_uc1, in_uc1] * split[in1_uc2, out2_uc2, out3_uc2, in_uc2] end
        
    @tensor begin horizontal_term_2[(out_uc1, out_uc2);(in_uc1,in_uc2)] := fuse[out_uc1, out1_uc1, out2_uc1, out3_uc1] * fuse[out_uc2, out1_uc2, out2_uc2, out3_uc2] * 
                                                                horizontal_term[out3_uc1, out2_uc2, in3_uc1, in2_uc2] *
                                                                split[out1_uc1, out2_uc1, in3_uc1, in_uc1] * split[out1_uc2, in2_uc2, out3_uc2, in_uc2] end    

    horizontal_term_coarse = horizontal_term_1 + horizontal_term_2

    #=
    now we create the vertical terms on the coarse grained level.
    the horizontal term consists of:
    1. A hopping term from site 1 of the UC to site 2 of the UC below.
    2. A hopping term from site 2 of the UC to site 3 of the UC below.
    =#

    @tensor begin vertical_term_1[(out_uc1, out_uc2);(in_uc1,in_uc2)] := fuse[out_uc1, out1_uc1, out2_uc1, out3_uc1] * fuse[out_uc2, out1_uc2, out2_uc2, out3_uc2] * 
        vertical_term[out1_uc1, out2_uc2, in1_uc1, in2_uc2] *
        split[in1_uc1, out2_uc1, out3_uc1, in_uc1] * split[out1_uc2, in2_uc2, out3_uc2, in_uc2] end

    @tensor begin vertical_term_2[(out_uc1, out_uc2);(in_uc1,in_uc2)] := fuse[out_uc1, out1_uc1, out2_uc1, out3_uc1] * fuse[out_uc2, out1_uc2, out2_uc2, out3_uc2] * 
        vertical_term[out2_uc1, out3_uc2, in2_uc1, in3_uc2] *
            split[out1_uc1, in2_uc1, out3_uc1, in_uc1] * split[out1_uc2, out2_uc2, in3_uc2, in_uc2] end    

    vertical_term_coarse = vertical_term_1 + vertical_term_2
                                                   
    ham_term_coarse_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(ham_term_coarse_arr, (loc = local_term_coarse, hor = horizontal_term_coarse, vert = vertical_term_coarse))
    end

return ham_term_coarse_arr
end