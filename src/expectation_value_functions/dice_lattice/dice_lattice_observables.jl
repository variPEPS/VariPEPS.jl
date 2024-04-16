function observables_Kitaev_Γ_model_dice(ham_parameters, loc, pattern_arr; Space_type = ℂ)

    #as a first, we look at the number operator and at the variance

    dim_square = dim(domain(loc[1,1])[3]) #local dimension of the square lattice tensor
    dim_dice = Int(cbrt(dim_square)) #local dimension of the d.o.f. on the dice lattice
    
    split = isomorphism(Space_type^dim_dice ⊗ Space_type^dim_dice ⊗ Space_type^dim_dice, Space_type^dim_square)
    fuse = isomorphism(Space_type^dim_square, Space_type^dim_dice ⊗ Space_type^dim_dice ⊗ Space_type^dim_dice)

    #here we build the creation an annihilation operators for the truncated local Hilbert spoace of the Bosons
    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = ℂ);

    #=
    here we create an array of the local terms that we want to evaluate for every site in the unit cell.
    =#

    @tensor σ_x_term_1a[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_x[out1,in1] * split[in1, out2, out3, in]
    @tensor σ_x_term_1b[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_x[out2,in2] * split[out1, in2, out3, in]
    @tensor σ_x_term_1c[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_x[out3,in3] * split[out1, out2, in3, in]

    @tensor σ_y_term_1a[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_y[out1,in1] * split[in1, out2, out3, in]
    @tensor σ_y_term_1b[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_y[out2,in2] * split[out1, in2, out3, in]
    @tensor σ_y_term_1c[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_y[out3,in3] * split[out1, out2, in3, in]

    @tensor σ_z_term_1a[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_z[out1,in1] * split[in1, out2, out3, in]
    @tensor σ_z_term_1b[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_z[out2,in2] * split[out1, in2, out3, in]
    @tensor σ_z_term_1c[(out,);(in,)] := fuse[out, out1, out2, out3] * σ_z[out3,in3] * split[out1, out2, in3, in]

    local_obs = [σ_x_term_1a, σ_x_term_1a, σ_x_term_1a, σ_y_term_1a, σ_y_term_1b, σ_y_term_1c, σ_z_term_1a, σ_z_term_1b, σ_z_term_1c]
                         
    local_obs_coarse_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(local_obs_coarse_arr, (loc = local_obs,))
    end

    return local_obs_coarse_arr
end
