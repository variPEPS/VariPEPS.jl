function ctmrg(loc_d_in, Bond_env, Pattern_arr; maxiter = 400, ϵ = 1e-7 , Space_type = ℝ, Projector_type = :half, Ham_parameters = nothing,
     model = :Heisenberg_square, calc_energy = true, observ = false, lattice = :square, identical = false, conv_info = false, sym_tensors = false, 
     D = :nothing, Dphys = :nothing, reuse_envs = false, output_envs = false, svd_type = :envbond, adjust_χ = false, spiral = false)
    

    if spiral == false
        u_paras = false
    else
        u_paras = loc_d_in[2]
        loc_d_in = loc_d_in[1]
    end

    if sym_tensors != false
        #as a first try hardcode the d = 3 case:
        loc_d_in = get_symmetric_tensors(loc_d_in, D, Dphys, sym_tensors)
    end

    #=
    The function CTMRG needs as an Input:
    1. An array of PEPS-tensors (5 dimensional arrays). One PEPS-tensor for each
    independent tensor in the unit cell. 

    2. An number that specifies the environment bond dimension of the environment tensors that are generated

    3. An array with the structure of the unit cell. This array needs to be two dimensional.
    The array needs numbers to specify the unit cell structure.
    e.g. for an unit cell with a Neél pattern one should specify:
    Pattern_arr = Array{Any}(undef,2,2)
    Pattern_arr[1,1] = 1
    Pattern_arr[2,1] = 2
    Pattern_arr[1,2] = 2
    Pattern_arr[2,2] = 1
    =#

    #it might be that we supply the input already in form of an array of TensorMaps - in this case we don't need to convert them! 
    if loc_d_in[1] isa TensorMap
        #in this case we already gave TensorMaps into the algorithm and do not need to convert them too such.
        PEPS_arr = convert_input(loc_d_in, Space_type = Space_type, lattice = lattice, identical = identical, inputisTM = true)  
    else
    #here we convert the input into tensor maps and map the original lattice to the square lattice.
        PEPS_arr = convert_input(loc_d_in, Space_type = Space_type, lattice = lattice, identical = identical)  
    end

    #Here we create an array of pointers, in the shape of the Pattern_arr that points to the relevant tensors in the PEPS_arr
    loc_d = pattern_function(PEPS_arr, Pattern_arr)
    #Create the adjoint "or Bra" PEPS tensors, and let this be ignores by the AD-engine, such that they behave as independent Tensors
    loc = Zygote.@ignore copy(loc_d)
    
    #now we need to initialize the environment arr.
    if reuse_envs == false
        env_arr = ini_multisite(loc, loc_d, Pattern_arr, PEPS_arr; Space_type = Space_type) 
    else 
        env_arr = reuse_envs
    end
        
    S_test_array_old = 0
    S_test_array2_old = 0
    iterations = 0
    env_arr_old = 0
    number = 10
    
    converging = true
    for i in 1:maxiter
        if conv_info 
            println("CTM-RG iteration $i")
        end
        
        S_test_array, S_test_array2, env_arr = CTMRG_step(env_arr, loc, loc_d, Bond_env, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
     
        #this is just some convergence check based on the SV of the environment tensors. 
        #For the calculation of the gradient we check for element wise convergence.
        if  i>2 && maximum(abs.(S_test_array - S_test_array_old)) < ϵ && maximum(abs.(S_test_array2 - S_test_array2_old)) < ϵ  
            break
        end
        
        env_arr_old = env_arr
       
        #this just prints some convergence info in case it is wanted.
        if i>2
            if conv_info
                println("this shows the convergence of two environment tensors")
                println(maximum(abs.(S_test_array - S_test_array_old)))
                println(maximum(abs.(S_test_array2 - S_test_array2_old)))
            end
        end
                 
        S_test_array_old = S_test_array
        S_test_array2_old = S_test_array2
        iterations = i
        
        if i == maxiter
            @info "CTMRG did not converge after maxiter = $(maxiter) steps."
            converging = false
        end
            
    end
    
    if converging == false
        return 10
    end
    
    #one can dynamically increase the bond dimension - this is done here
    if adjust_χ != false
        trunc_sv_arr, env_arr = CTMRG_step(env_arr, loc, loc_d, Bond_env, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_check = true)

        if maximum(trunc_sv_arr) > adjust_χ[1] #if the largest SV cut during generation of the projectors is larger than the threshhold value increase chi
            while Bond_env < adjust_χ[2] && maximum(trunc_sv_arr) > adjust_χ[1]
                @info "the environment bond dimension is being increased from $(Bond_env) to $Bond_env+2"
                Bond_env = Bond_env + 2
                trunc_sv_arr, env_arr = CTMRG_step(env_arr, loc, loc_d, Bond_env, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_check = true)
                test = maximum(trunc_sv_arr)
                @info "the largest truncation error is $(test)"
            
            end
            if Bond_env ≥ adjust_χ[2]
                @info "the maximal allowed environment bond dimension was reached"
            else
                @info "the SV at which we truncate is now smaller than the threshhold!"
            end
        else
            @show maximum(trunc_sv_arr)
        end
    end

    output = []

    if calc_energy == true
        energy_density = calculate_energy_density(env_arr, loc, loc_d, Pattern_arr; Space_type = Space_type, Ham_parameters = Ham_parameters, model = model, lattice = lattice, u_paras = u_paras)
        append!(output, energy_density)
    end

    if observ == true
        observables = calculate_observables(env_arr, loc, loc_d, Pattern_arr; Space_type = Space_type, Ham_parameters = Ham_parameters, model = model, lattice = lattice)
        append!(output, [observables])
    end

    if output_envs == true
        append!(output, env_arr)
    end

    return output
end



