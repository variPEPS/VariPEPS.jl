function rank(A::TensorMap)
    rank_dom = length(dims(codomain(A)))
    rank_codom = length(dims(domain(A)))
    
    return rank_dom + rank_codom
end

function sqrtTM(S)
    Smat = convert_TM_to_mat(S)
    #display(typeof(Smat))
    B = sqrt.(diag(Smat))
    Ssqrt = TensorMap(diagm(B), codomain(S), domain(S))
    return Ssqrt
end

function p_arr_inv(a, tol)
    
    N = maximum(a)
    
    res = [el ≥ tol*N ? 1/el : 0.0 for el in a]
    
    return res
end



function pinv_sqrt(S, tol)
    #Smat = S.data
    
    #this awkward way was somehow nessessary - will be changed in future.
    S_dict = convert(Dict,S)
    A = S_dict[:data]["Trivial()"]
    Smat = A*Matrix(I,size(A)[1],size(A)[1])
    #Smat = A
    #display(size(Smat))
    
    B = sqrt.(diag(Smat))
    C = p_arr_inv(B, tol)
    F = diagm(C)
    #display(size(C))
    S_inv_sqrt = TensorMap(F, codomain(S) ← domain(S))
    return S_inv_sqrt
end



function convert_TM_to_mat(A)
    Adict = convert(Dict, A)
    AMat = Adict[:data]["Trivial()"]
    return AMat
end


function apply_mat(x, L)
    b = Tensor(x, domain(L))
    c = L * b
    return c.data[:,1]
end

function apply_mat_adj(x, L)
    b = Tensor(x, domain(L'))
    c = L' * b
    return c.data[:,1]
end

function tsvd_GKL(A; χ::Int = 20, space_type = ℂ)
                
    b = rand(dim(codomain(A)))
                
    f_A = x -> apply_mat(x, A)
    f_A_adj = x -> apply_mat_adj(x, A)            
    #display("update")
    S_kr, U_kr, V_kr, info = svdsolve((f_A, f_A_adj), b, χ, :LR, krylovdim = 2*χ)
                
    #put some warning here if the GKL did not work.
    if χ > info.converged
        @warn "here not the GKL procedure was not able to converge the desired number of singular values! the SVD is now calculated by conventional means..."
        #@info "number of Krylov-subspace restarts, number of operations with the linear map, list of residuals" info.numiter info.numops info.normres

        #@info "the SVD is now calculated by conventional means..."
        U, S, Vd = tsvd(A, trunc = truncspace(space_type^χ), alg = TensorKit.SDD())
        return U, S, Vd

    end
    
    U_mat = U_kr[1]
    for i in 2:χ     
        U_mat = hcat(U_mat, U_kr[i])
    end
    U_TM = TensorMap(U_mat, codomain(A) ← space_type^χ)

    V_mat = adjoint(V_kr[1])
    for i in 2:χ
        V_mat = vcat(V_mat, adjoint(V_kr[i]))
    end
    V_d_TM = TensorMap(V_mat, space_type^χ ← domain(A))

    S_TM = TensorMap(diagm(S_kr[1:χ]), space_type^χ ← space_type^χ)
                    
    return U_TM, S_TM, V_d_TM
end


#from TensorKitAD
function _elementwise_mult(a::AbstractTensorMap,b::AbstractTensorMap)
    dst = similar(a);
    for (k,block) in blocks(dst)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    dst
end

function unique_tsvd(A, Bond_env; Space_type = ℝ, svd_type = :GKL, split = :yes)

    if svd_type == :accuracy
        U, S, Vd = tsvd(A, trunc = notrunc(), alg = TensorKit.SDD())

    elseif svd_type == :envbond

        U, S, Vd = tsvd(A, trunc = truncspace(Space_type^Bond_env), alg = TensorKit.SDD())
    
    elseif svd_type == :GKL

        U, S, Vd = tsvd_GKL(A, χ = Bond_env, space_type = Space_type)
            
    else
        display("you have not specified a truncation type in function: unique_svd")                
    end

    #display("SDD is used now!")
    

    #be careful- when we include symmetries we have to go through the symmetry sectors indvidually
    UMat = convert_TM_to_mat(U)
    VdMat = convert_TM_to_mat(Vd)
    
    #here we fix the gauge for elementwise convergence of the environment tensors
    if Space_type == ℝ
        absmax = x -> abs(minimum(x)) > abs(maximum(x)) ? minimum(x) : maximum(x)
        index_comp = x -> findmax(x)[2] > findmin(x)[2] ? minimum(x) : maximum(x)
        descide_func = x -> isapprox(abs(minimum(x)), abs(maximum(x)); rtol = 1e-8) ? index_comp(x) : absmax(x)
        fix_mat = diagm(sign.(map(descide_func, eachcol(UMat))))
    end
    
    if Space_type == ℂ
        absmax = x -> x[partialsortperm(abs.(x), 1:2; rev = true)][1]
        index_comp = x -> partialsortperm(abs.(x), 1:2; rev = true)[2] > partialsortperm(abs.(x), 1:2; rev = true)[1] ? x[partialsortperm(abs.(x), 1:2; rev = true)][1] : x[partialsortperm(abs.(x), 1:2; rev = true)][2]
        descide_func = x -> isapprox(abs.(x)[partialsortperm(abs.(x), 1:2; rev = true)][1] , abs.(x)[partialsortperm(abs.(x), 1:2; rev = true)][2] ; rtol = 10^-8) ? index_comp(x) : absmax(x)
        fix_mat = diagm(exp.(-angle.(map(descide_func, eachcol(UMat)))*im))
    end
    
    Ufixed1 = UMat*fix_mat
    Vdfixed1 = fix_mat' * VdMat
    if split == :yes
        Ufixed = reshape(Ufixed1, (dim(codomain(U)[1]), dim(codomain(U)[2]), dim(codomain(U)[3]), dim(domain(U))))
        Vdfixed = reshape(Vdfixed1, (dim(codomain(Vd)), dim(domain(Vd)[1]), dim(domain(Vd)[2]), dim(domain(Vd)[3]) ))

        UfTensor = TensorMap(Ufixed, codomain(U) ← domain(U))
        VdfTensor = TensorMap(Vdfixed, codomain(Vd) ← domain(Vd))
        
    elseif split == :no
        UfTensor = TensorMap(Ufixed1, codomain(U) ← domain(U))
        VdfTensor = TensorMap(Vdfixed1, codomain(Vd) ← domain(Vd))
        
    end
        
    return UfTensor , S , VdfTensor
end

#old and ugly function that initializes the environment tensors.
function ini_multisite(loc, loc_d, Pattern_arr, loc_arr; Space_type = Space_type)
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    
    env_arr = Array{Any}(undef, size(loc_arr)[1])
    
    donealready = Dict()

    for i in 1:Lx, j in 1:Ly
        
        #check that we do not create an extra environment for tensors that appear multiple times in the unit cell
        if Pattern_arr[i,j] in keys(donealready)
            
            continue
        
        end
        
        
        
        # m is i-1
        if i==1
            m = Lx
        else 
            m = i-1
        end
        
        # n is i + 1
        if i == Lx
            n = 1
        else 
            n = i+1
        end
        
        # k is j-1
        if j == 1
            k = Ly
        else 
            k = j-1
        end
        
        #l is j+1
        if j == Ly
            l = 1
        else 
            l = j+1
        end
        
        
        #Dim_loc = dim(domain(loc[i,j])[5])
        Bond_loc = dim(codomain(loc[i,j])[1])

        space_virt = Space_type^Bond_loc
        #space_loc = Space_type^Dim_loc
        
        Fuse = isomorphism(fuse(dual(space_virt) ⊗ space_virt), dual(space_virt) ⊗ space_virt)
        Unfuse = isomorphism(dual(space_virt) ⊗ space_virt, fuse(dual(space_virt) ⊗ space_virt))


        #initialization of environment tensors as random tensors in hermitian fashion
        @tensor C_ul[(i,);(j,)] := Fuse[i,i1,i2] * loc[m,k][v1,i2,j2,v2,a] * conj(loc[m,k][v1,i1,j1,v2,a]) * Unfuse[j1,j2,j] 

        @tensor C_ur[(i,j);()] := Fuse[i,i1,i2] * Fuse[j,j1,j2] * loc[n,k][i2,j2,v1,v2,a]*conj(loc[n,k][i1,j1,v1,v2,a])

        @tensor C_dl[();(i,j)] := loc[m,l][v1,v2,i2,j2,a] * conj(loc[m,l][v1,v2,i1,j1,a]) * Unfuse[i1,i2,i] * Unfuse[j1,j2,j]

        @tensor C_dr[(i,);(j,)] := Fuse[i,i1,i2] * loc[n,l][i2,v1,v2,j2,a]*conj(loc[n,l][i1,v1,v2,j1,a]) * Unfuse[j1,j2,j]


        @tensor Tr_u[(i,j1,j2);(k)] := Fuse[i,i1,i2] * loc[i,k][i2,j2,k2,v2,a]*conj(loc[i,k][i1,j1,k1,v2,a]) * Unfuse[k1,k2,k]

        @tensor Tr_r[(i1,i2,j);(k)] := Fuse[j,j1,j2]* loc[n,j][i2,j2,v1,k2,a]*conj(loc[n,j][i1,j1,v1,k1,a])  * Unfuse[k1,k2,k]

        @tensor Tr_d[(i,);(j,k1,k2)] := Fuse[i,i1,i2] * loc[i,l][i2,v2,j2,k2,a]*conj(loc[i,l][i1,v2,j1,k1,a]) * Unfuse[j1,j2,j] 

        @tensor Tr_l[(i,);(j1,j2,k)] := Fuse[i,i1,i2] * loc[m,j][v2,i2,j2,k2,a]*conj(loc[m,j][v2,i1,j1,k1,a])  * Unfuse[k1,k2,k]

        environment = C_ul, C_ur, C_dl, C_dr, Tr_u, Tr_r, Tr_d, Tr_l
        
        C_ul1 = normalization_convention(C_ul)
        C_ur1 = normalization_convention(C_ur)
        C_dl1 = normalization_convention(C_dl)
        C_dr1 = normalization_convention(C_dr)
        Tr_u1 = normalization_convention(Tr_u)
        Tr_r1 = normalization_convention(Tr_r)
        Tr_d1 = normalization_convention(Tr_d)
        Tr_l1 = normalization_convention(Tr_l)

        env_arr[Pattern_arr[i,j]] = (ul = C_ul1, ur = C_ur1, dl = C_dl1, dr = C_dr1,
                                        u = Tr_u1, r = Tr_r1, d = Tr_d1, l = Tr_l1)
        
        donealready[Pattern_arr[i,j]] = 1
        
    end
    return env_arr
end

function initialize_PEPS(Bond_loc, Dim_loc, Number_of_PEPS; Number_type = Float64, lattice = :square, identical = false, seed = 1236, data_type = :TensorMap, Space_type = ℂ)
    #=
    NOTE1: 
    The number of tensors refers to the number of coarse grained tensors that result on the square lattice!
    For example if we are working fundamentally with a model on a honeycomb lattice we thus need an array with 2*Number_of_PEPS tensors as input!
    
    NOTE2:
    "Dim_loc" can refer to the dimension of different spaces. If we work with the d.o.f. on the original lattice before coarse graining,
    it refers to the dimension of the local Hilbert space of that lattice.
    E.g. for a spin 1/2 model on the honeycomb lattice we would have Dim_loc = 2 . 
    However if we want/need to optimize directly on the coarse grained lattice it refers to the product of the local Hilbert spaces
    that are encompassed by the coarse grained tensor. 
    E.g. for a spin 1/2 model on the dice lattice- where we optimize on the coarse grained square lattice tensors- we would have Dim_loc = 8 = 2*2*2
    =#

    loc_in = []
    rng = MersenneTwister(seed)
    
    space_virt = Space_type^Bond_loc
    space_loc = Space_type^Dim_loc

    randn_with_seed = (tuple) -> randn(rng, Number_type, tuple)

    #generically the input will be TensorMaps---> the alternative will be removed at some point.
    if data_type == :TensorMap

        if lattice == :square

            for i in 1:Number_of_PEPS
                input_tensor = TensorMap(randn_with_seed, space_virt ⊗ space_virt  ←  space_virt ⊗ space_virt ⊗ (space_loc)')
                push!(loc_in, normalization_convention(input_tensor))
            end
            
        elseif lattice == :honeycomb

            if identical == false

                for i in 1:Number_of_PEPS

                    if i%2 == 1
                        #make a TensorMap out of the Arrays in loc_in
                        input_tensor = TensorMap(randn_with_seed, Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc)')
                        push!(loc_in, normalization_convention(input_tensor))

                    else

                        input_tensor = TensorMap(randn_with_seed, Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc  ⊗ (Space_type^Dim_loc)')
                        push!(loc_in, normalization_convention(input_tensor))

                    end

                end
            
            end

            if identical == true

                for i in 1:Number_of_PEPS 

                    loc_in_data = randn_with_seed((Bond_loc,Bond_loc,Bond_loc,Dim_loc))
                    loc_in_data_twisted = permutedims(loc_in_data, (2,3,1,4))
                    #make a TensorMap out of the Arrays in loc_in
                    input_tensor = TensorMap(loc_in_data, Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc)')
                    push!(loc_in, normalization_convention(input_tensor))

                    input_tensor = TensorMap(loc_in_data_twisted, Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc  ⊗ (Space_type^Dim_loc)')
                    push!(loc_in, normalization_convention(input_tensor))

                end
                
            end

        elseif lattice == :dice
            #=note here, that if we were to create three tensors (one for every site in the UC of the dice lattice),
            we run into the problem that the resulting coarse grained tensor has a bond dimenison of d^2 where d is the bond dimension of the 
            tensors of on the dice lattice. This only then leads to very few bond dimensions that are feasable.
            Hence we take as the input tensors, aka the parameters we optimize on a already coase grained - square lattice- tensor with
            an increased bond dimension.=#
            for i in 1:Number_of_PEPS
                input_tensor = TensorMap(randn_with_seed, space_virt ⊗ space_virt  ←  space_virt ⊗ space_virt ⊗ (space_loc)')
                push!(loc_in, normalization_convention(input_tensor))
            end
        elseif lattice == :kagome
            #=here a number of elegant mappings (e.g.: Kagome -> Triangle -> Honeycomb -> Square) are in principle possible.
            as a first approach however we just put a square lattice PEPS one the Kagome-lattice.=#
            for i in 1:Number_of_PEPS
                input_tensor = TensorMap(randn_with_seed, space_virt ⊗ space_virt  ←  space_virt ⊗ space_virt ⊗ (space_loc)')
                push!(loc_in, normalization_convention(input_tensor))
            end
        end
    else

        if lattice == :square

            for i in 1:Number_of_PEPS
                push!(loc_in, normalization_convention(randn(rng, Number_type, Bond_loc, Bond_loc, Bond_loc, Bond_loc, Dim_loc)))
            end
            
        elseif lattice == :honeycomb

            if identical == false
                for i in 1:2*Number_of_PEPS
                    push!(loc_in, normalization_convention(randn(rng, Number_type, Bond_loc, Bond_loc, Bond_loc, Dim_loc)))
                end
            end

            if identical == true
                for i in 1:Number_of_PEPS
                    push!(loc_in, normalization_convention(randn(rng, Number_type, Bond_loc, Bond_loc, Bond_loc, Dim_loc)))
                end
            end

        else
            println("you have not defined this lattice in the function 'initialize_PEPS'")
        end
    
    end
        
        
    return loc_in
end

function pattern_function(arr_in, Pattern_arr)
    buf = Buffer(arr_in, size(Pattern_arr))
    for i in 1:size(Pattern_arr)[1], j in 1:size(Pattern_arr)[2]
        buf[i,j] = arr_in[Pattern_arr[i,j]]
    end
    return copy(buf)
end

function normalization_convention(A; fix_phase = true)
    N = Zygote.@ignore norm(A) 
    A1 = (2*A/N)

    #when the data is real, don't do the rotation! Can be done with keyword as well.
    if A isa TensorMap
        if spacetype(A) == CartesianSpace
            return A1
        end
    elseif isreal(A)
        return A1
    end

    if fix_phase == false #
        return A1
    end
    if A isa TensorMap
        absmax = Zygote.@ignore A.data[findmax(abs.(A.data))[2]]
    else 
        absmax = Zygote.@ignore A[findmax(abs.(A))[2]]
    end
    phase_absmax = angle(absmax)
    N2 = exp(-phase_absmax * im)
    A2 = (A1*N2)
    return A2
end

function normalization_convention_without_phase(A)
    N = Zygote.@ignore norm(A) 
    A1 = (A/N)
    return A1 , N
end


function convert_input(loc_in; lattice = :square, Space_type = ℝ, identical = false, inputisTM = false)    
    
    #=
    In this function we take the input as an array, which is needed for the optimizer, and convert it 
    into TensorMaps (WHICH WE NORMALIZE). These TensorMaps are then an array. This functions is different based on the lattice of the 
    model considered. One should add a specific procedure for all new lattices that we implement.
    =#
    
    if lattice == :square
        if inputisTM == true
            return loc_in
        end

        #infer Bond-dimension of the PEPS-tensor and local Hilbert-Space dimension from the local PEPS tensor
        Bond_loc = size(loc_in[1])[1]
        Dim_loc = size(loc_in[1])[5]

        #buf = Buffer([], TensorMap, size(loc_in)[1])
        buf = Buffer([], TensorMap, length(loc_in))

        #for i in 1:size(loc_in)[1]
        for i in 1:length(loc_in)   
            #make a TensorMap out of the Arrays in loc_in
            buf[i] = TensorMap( normalization_convention( loc_in[i]), Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc)')
        end

    elseif lattice == :triangular
        if inputisTM == true
            return loc_in
        end

    elseif lattice == :honeycomb
        #= when inputting tensors for the honeycomb lattice, one should supply them in the following form:
        every tensor in the unit cell of our square lattice, that we have mapped the honeycomb lattice onto corrsponds to two Tensors on the 
        honeycomb lattice. These two tensors are inequivalent (at least that is what I am thinking right now). For a unit cell with n-sites 
        one should thus supply an input array of 2n tensors of rank 4 - they each have 3 virtual indices and one physical index. This function than contracts 
        two adjacent tensors to form the coarse grained tensors that live on the square lattice that we perform the CTM-RG on.
        =#        

        if inputisTM == false
            #infer Bond-dimension of the honeycomb-PEPS-tensor and local Hilbert-Space dimension from the local honeycomb-PEPS tensor
            Bond_loc = size(loc_in[1])[1]
            Dim_loc_honey = size(loc_in[1])[4] 

            #=We can motivated by the bond structure of the Kitaev model enforce that the two honeycomb tensors that get coarse grained to a
            single square lattice tensor are related to each other by rotation... This has been implemented here, if one chooses the keyword "identical" to
            be true. =#
            if identical == true
                loc_in_TM = convert_loc_in_to_TM(loc_in, lattice = :honeycomb, Space_type = Space_type, identical = true)
            else
                loc_in_TM = convert_loc_in_to_TM(loc_in, lattice = :honeycomb, Space_type = Space_type)

            end
        else 
            Bond_loc = dim(space(loc_in[1])[1])
            Dim_loc_honey = dim(space(loc_in[1])[4]) 

            
            loc_in_TM = loc_in
            
        end

        Fuse = isomorphism(fuse(Space_type^Dim_loc_honey ⊗ Space_type^Dim_loc_honey), Space_type^Dim_loc_honey ⊗ Space_type^Dim_loc_honey)
        
        buf = Buffer([], TensorMap, Int(length(loc_in_TM)/2))
        
        for (i,j) in enumerate(1:2:length(loc_in_TM))
            
            @tensor sqr_lat_tens[(v1,v2);(v3,v4,p)] := loc_in_TM[j][v1,c,v4,p1] * loc_in_TM[j+1][c,v2,v3,p2] * Fuse[p,p1,p2]
            
            sqr_lat_tens_norm = normalization_convention(sqr_lat_tens)
            
            buf[i] = sqr_lat_tens
            
        end    

    elseif lattice == :dice

        if inputisTM == true
            return loc_in
        end
    elseif lattice ==:kagome

        if inputisTM == true
            return loc_in
        end
        
    else
        println("the lattice you have chosen is not yet included in the function 'convert_input'! You should add it!")
        
    end

    
    return copy(buf)
end

function convert_loc_in_to_TM(loc_in; lattice = :honeycomb, Space_type = ℝ, identical = false)
    if lattice == :honeycomb
        #infer Bond-dimension of the PEPS-tensor and local Hilbert-Space dimension from the local PEPS tensor
        Bond_loc = size(loc_in[1])[1]
        Dim_loc_honey = size(loc_in[1])[4]

        if identical == false

            buf = Buffer([], TensorMap, length(loc_in))

            for i in 1:length(loc_in)   

                if i%2 == 1
                    #make a TensorMap out of the Arrays in loc_in
                    buf[i] = TensorMap( normalization_convention( loc_in[i] ), Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc_honey)')
                
                else

                    buf[i] = TensorMap( normalization_convention( loc_in[i] ), Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc  ⊗ (Space_type^Dim_loc_honey)')
                end

            end
        
        end

        if identical == true

            buf = Buffer([], TensorMap, 2 * length(loc_in))

            for i in 1:length(loc_in)   

                turned_loc_in = permutedims(loc_in[i], (2,3,1,4))

                #make a TensorMap out of the Arrays in loc_in
                buf[2*i-1] = TensorMap( normalization_convention( loc_in[i] ), Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc_honey)')
                
                buf[2*i] = TensorMap( normalization_convention( turned_loc_in ), Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc  ⊗ (Space_type^Dim_loc_honey)')
                
            end
            
        end


    end
    return copy(buf)
end

function test_elementwise_convergence(env_arr, env_arr_old, Pattern_arr, conv_crit)
    
    #this counts the number of environment-tensors that are NOT YET converged element wise.
    number = 0
    
   
    value_of_conv_max = 0
    value_of_conv_min = 0

    
    for i in 1:size(env_arr)[1]
        comp_ul = env_arr[i].ul - env_arr_old[i].ul 
        #this is converged element-wise if every element in the mask "comp_mask_..." is 1
        comp_mask_ul = abs.(comp_ul.data) .< conv_crit
        
        comp_dl = env_arr[i].dl - env_arr_old[i].dl 
        comp_mask_dl = abs.(comp_dl.data) .< conv_crit
        
        comp_dr = env_arr[i].dr - env_arr_old[i].dr 
        comp_mask_dr = abs.(comp_dr.data) .< conv_crit
        
        comp_ur = env_arr[i].ur - env_arr_old[i].ur 
        comp_mask_ur = abs.(comp_ur.data) .< conv_crit
        
        comp_u = env_arr[i].u - env_arr_old[i].u 
        comp_mask_u = abs.(comp_u.data) .< conv_crit
        
        comp_r = env_arr[i].r - env_arr_old[i].r 
        comp_mask_r = abs.(comp_r.data) .< conv_crit
        
        comp_d = env_arr[i].d - env_arr_old[i].d 
        comp_mask_d = abs.(comp_d.data) .< conv_crit
        
        comp_l = env_arr[i].l - env_arr_old[i].l 
        comp_mask_l = abs.(comp_l.data) .< conv_crit
        

        
        if minimum(comp_mask_ul) == 0 
            #=
            if the minimum of the mask "comp_mask_..." is 0 that means 
            that there are elements that are NOT YET converged element wise
            --> Thus add 1 to the number of these tensors
            =#
            number += 1
        
            #a = norm(comp_ul)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_dl) == 0 
            number += 1
            
            #a = norm(comp_dl)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_dr) == 0 
            number += 1
            
        
            #a = norm(comp_dr)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_ur) == 0 
            number += 1
        
            #a = norm(comp_ur)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_u) == 0 
            number += 1
        
            #a = norm(comp_u)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_r) == 0 
            number += 1
        
            #a = norm(comp_r)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_d) == 0 
            number += 1
        
            #a = norm(comp_d)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_l) == 0 
            number += 1
        
            #a = norm(comp_l)
            #display("the norm difference is $a")
        end
        
        
    end

    
    return number
end

function make_lin_array(A)
    return ArrayPartition(A...)
end

function get_symmetric_tensors(parameter_array, D, Dphys, sym_tensors)

    if sym_tensors == :R_PT
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_R_and_PT_sym(D,Dphys)
    elseif sym_tensors == :R
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_R_sym(D,Dphys)
    elseif sym_tensors == :PT
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_sym(D,Dphys)
    elseif sym_tensors == :P
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_P_sym(D,Dphys)
    elseif sym_tensors == :P_minus
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_P_minus_sym(D,Dphys)
    elseif sym_tensors == :PT_phase
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_sym(D,Dphys)
    elseif sym_tensors == :PT_test
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_sym(D,Dphys)
    elseif sym_tensors == :PT_minus
        symmetric_tensors_imag, symmetric_tensors_real = Zygote.@ignore create_sym_tensors_PT_sym(D,Dphys)
    elseif sym_tensors == :PT_2
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_sym_2(D,Dphys)    
    elseif sym_tensors == :PT_I
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_and_I_sym(D,Dphys)
    else
        println("you have not implemented the local symmetry that you ask of the tensor")
    end
    #display(symmetric_tensors_real ≈ real(symmetric_tensors_real))
    #display(symmetric_tensors_imag ≈ real(symmetric_tensors_imag))
    N_real = size(symmetric_tensors_real)[2]
    N_imag = size(symmetric_tensors_imag)[2]


    buf = Zygote.Buffer([], Array, length(parameter_array))

    #loc_d_in = []
    for i in 1:length(parameter_array)
        
        loc_d_in_vec = zeros(size(symmetric_tensors_real)[1])
        if sym_tensors == :PT_test
            for j in 1:N_real
                loc_d_in_vec += parameter_array[i][j] * symmetric_tensors_real[:,j]
            end
            for j in 1:N_imag
                loc_d_in_vec += parameter_array[i][N_real + j] * symmetric_tensors_imag[:,j]
            end
        else
            for j in 1:N_real
                loc_d_in_vec += parameter_array[i][j] * symmetric_tensors_real[:,j]
            end
            for j in 1:N_imag
                loc_d_in_vec += im * parameter_array[i][N_real + j] * symmetric_tensors_imag[:,j]
            end
        end

        if sym_tensors == :PT_phase
            phase = parameter_array[i][end] / norm(parameter_array[i][end])
            loc_d_in_vec_phase = phase .* loc_d_in_vec
            #display(loc_d_in_vec_phase)
        else 
            loc_d_in_vec_phase = loc_d_in_vec
        end

        buf[i] = reshape(loc_d_in_vec_phase, D, D, D, D, Dphys)
        
    end

        

    return copy(buf)
end

function create_sym_tensors_R_and_PT_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)

    #create the rotation operator
    @tensor R[δ,α,β,γ,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape the rotation operator into a matrix
    R_mat = reshape(R, D^4*Dphys, D^4*Dphys);
    #calculate the eigenvalues/vectors of the Rotation matrix
    values_R, vectors_R = eigen(R_mat);
    #take only those eigenvectors that have eigenvalue 1 -> create the appropriate mask for this purpose
    mask_R = [values_R[i] ≈ 1 ? true : false for i in eachindex(values_R)];
    #create the reflexion-operator
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);
    #look at the reflection matrix in the subspace of vectors that span the eigenspace of R corresponding to eigenvalue 1
    Rx_mat_sub = vectors_R'[mask_R,:] * Rx_mat * vectors_R[:,mask_R];
    #look at the eigenvalues and eigenvectors in this subspace
    values_Rx, vectors_Rx = eigen(Rx_mat_sub);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]
    #now we can concatenate these to get the tensors that have eigenvalue +1 for rotation and reflextion
    symmetric_tensors_real = vectors_R[:,mask_R] * vectors_Rx[:,mask_Rx_plus]
    #now we can concatenate these to get the tensors that have eigenvalue +1 for rotation and -1 for reflextion
    symmetric_tensors_imag = vectors_R[:,mask_R] * vectors_Rx[:,mask_Rx_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_R_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)

    #create the rotation operator
    @tensor R[δ,α,β,γ,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape the rotation operator into a matrix
    R_mat = reshape(R, D^4*Dphys, D^4*Dphys);
    #calculate the eigenvalues/vectors of the Rotation matrix
    values_R, vectors_R = eigen(R_mat);
    #take only those eigenvectors that have eigenvalue 1 -> create the appropriate mask for this purpose
    mask_R = [values_R[i] ≈ 1 ? true : false for i in eachindex(values_R)];

    symmetric_tensors_real = vectors_R[:,mask_R] 
    symmetric_tensors_imag = Matrix{Float64}(I,0,0)
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_PT_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]

    symmetric_tensors_real = vectors_Rx[:,mask_Rx_plus]
    symmetric_tensors_imag = vectors_Rx[:,mask_Rx_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_P_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]

    symmetric_tensors_real = vectors_Rx[:,mask_Rx_plus]
    symmetric_tensors_imag = []
    
    return symmetric_tensors_real, symmetric_tensors_imag
end


function create_sym_tensors_P_minus_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]

    symmetric_tensors_real = []
    symmetric_tensors_imag = vectors_Rx[:,mask_Rx_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_PT_sym_2(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator
    @tensor Rx[γ,β,α,δ,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]

    symmetric_tensors_real = vectors_Rx[:,mask_Rx_plus]
    symmetric_tensors_imag = vectors_Rx[:,mask_Rx_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_PT_and_I_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator on the x_axis
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]
    
    @tensor Ry[γ,β,α,δ,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Ry_mat = reshape(Ry, D^4*Dphys, D^4*Dphys);
    
    Ry_mat_sub_plus = vectors_Rx'[mask_Rx_plus,:] * Ry_mat * vectors_Rx[:,mask_Rx_plus];
    Ry_mat_sub_minus = vectors_Rx'[mask_Rx_minus,:] * Ry_mat * vectors_Rx[:,mask_Rx_minus];

    #look at the eigenvalues and eigenvectors in this subspace
    values_Ry_plus, vectors_Ry_plus = eigen(Ry_mat_sub_plus);
    mask_Ry_plus = [values_Ry_plus[i] ≈ 1 ? true : false for i in eachindex(values_Ry_plus)]
    
    values_Ry_minus, vectors_Ry_minus = eigen(Ry_mat_sub_minus);
    mask_Ry_minus = [values_Ry_minus[i] ≈ -1 ? true : false for i in eachindex(values_Ry_minus)]

    symmetric_tensors_real = vectors_Rx[:,mask_Rx_plus] * vectors_Ry_plus[:,mask_Ry_plus]
    symmetric_tensors_imag = vectors_Rx[:,mask_Rx_minus] * vectors_Ry_minus[:,mask_Ry_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end