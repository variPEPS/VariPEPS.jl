function use_Heisenberg_kagome_lattice(Ham_parameters, loc, pattern_arr; Space_type = ℂ)

    dim_square = dim(domain(loc[1,1])[3]) #local dimension of the square lattice tensor
    dim_dice = Int(cbrt(dim_square)) #local dimension of the d.o.f. on the dice lattice
    
    split = isomorphism(Space_type^dim_dice ⊗ Space_type^dim_dice ⊗ Space_type^dim_dice, Space_type^dim_square)
    fuse = isomorphism(Space_type^dim_square, Space_type^dim_dice ⊗ Space_type^dim_dice ⊗ Space_type^dim_dice)

    #here we build the spin 1/2 operators
    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = Space_type)

   
    h = Ham_parameters.h
    J = Ham_parameters.J
    dir = Ham_parameters.dir

    #in this way the tensor is real valued.
    @tensor Heisenberg_term[(i, j);(k, l)] :=  J * (σ_p[i,k] * σ_m[j, l] + σ_m[i,k] * σ_p[j,l])/2 + (σ_z[i,k] * σ_z[j,l])

    #add a local magnetic field if wanted
    h_term = h * (dir[1] *  σ_x + dir[2] * σ_y + dir[3] * σ_z)
    #=
    now we create the local term of the model on the coarse grained level. 
    The local term consists of as sum of:
    See Notes for definition of the unit cell.
    1. local terms for the three sites within the UC.
    2. Heisenberg term from the first to the second site in the unit cell.
    3. Heisenberg term from the second to the third site in the unit cell.
    4. Heisenberg term from the third to the first site in the unit cell.
    =#

    @tensor local_term_1a[(out,);(in,)] := fuse[out, out1, out2, out3] * h_term[out1,in1] * split[in1, out2, out3, in]
    @tensor local_term_1b[(out,);(in,)] := fuse[out, out1, out2, out3] * h_term[out2,in2] * split[out1, in2, out3, in]
    @tensor local_term_1c[(out,);(in,)] := fuse[out, out1, out2, out3] * h_term[out3,in3] * split[out1, out2, in3, in]

    @tensor local_term_2[(out,);(in,)] := fuse[out, out1, out2, out3] * Heisenberg_term[out1,out2,in1,in2] * split[in1, in2, out3, in]
    @tensor local_term_3[(out,);(in,)] := fuse[out, out1, out2, out3] * Heisenberg_term[out2,out3,in2,in3] * split[out1, in2, in3, in]
    @tensor local_term_4[(out,);(in,)] := fuse[out, out1, out2, out3] * Heisenberg_term[out3,out1,in3,in1] * split[in1, out2, in3, in]

    local_term_coarse = local_term_1a + local_term_1b + local_term_1c + local_term_2 + local_term_3 + local_term_4
    
    #=
    now we create the horizontal terms on the coarse grained level.
    the horizontal term consists of:
    1. A Heisenberg term from site 2 of the UC to site 3 of the UC to the right.
    =#
    
    #these contractions might need to be optimized!
    @tensor begin horizontal_term[(out_uc1, out_uc2);(in_uc1,in_uc2)] := fuse[out_uc1, out1_uc1, out2_uc1, out3_uc1] * fuse[out_uc2, out1_uc2, out2_uc2, out3_uc2] * 
                                                                Heisenberg_term[out2_uc1, out3_uc2, in2_uc1, in3_uc2] *
                                                                split[out1_uc1, in2_uc1, out3_uc1, in_uc1] * split[out1_uc2, out2_uc2, in3_uc2, in_uc2] end    

    horizontal_term_coarse = horizontal_term

    #=
    now we create the vertical terms on the coarse grained level.
    the horizontal term consists of:
    1. A Heisenberg term from site 3 of the UC to site 1 of the UC below.
    =#

    @tensor begin vertical_term[(out_uc1, out_uc2);(in_uc1,in_uc2)] := fuse[out_uc1, out1_uc1, out2_uc1, out3_uc1] * fuse[out_uc2, out1_uc2, out2_uc2, out3_uc2] * 
        Heisenberg_term[out3_uc1, out1_uc2, in3_uc1, in1_uc2] *
        split[out1_uc1, out2_uc1, in3_uc1, in_uc1] * split[in1_uc2, out2_uc2, out3_uc2, in_uc2] end


    vertical_term_coarse = vertical_term
                                                   
    #=
    now we create the diagonal terms on the coarse grained level.
    the diagonal term consists of:
    1. A Heisenberg term from site 2 of the UC to site 1 of the UC below and to the right.
    =#

    @tensor begin diagonal_term[(out_uc1, out_uc2);(in_uc1,in_uc2)] := fuse[out_uc1, out1_uc1, out2_uc1, out3_uc1] * fuse[out_uc2, out1_uc2, out2_uc2, out3_uc2] * 
        Heisenberg_term[out2_uc1, out1_uc2, in2_uc1, in1_uc2] *
        split[out1_uc1, in2_uc1, out3_uc1, in_uc1] * split[in1_uc2, out2_uc2, out3_uc2, in_uc2] end

    diagonal_term_coarse = diagonal_term

    ham_term_coarse_arr = []

    for i in minimum(pattern_arr):maximum(pattern_arr)
        push!(ham_term_coarse_arr, (loc = local_term_coarse, hor = horizontal_term_coarse, vert = vertical_term_coarse, diag = diagonal_term_coarse))
    end

return ham_term_coarse_arr
end