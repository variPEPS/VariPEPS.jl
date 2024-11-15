function converge_environment(loc, loc_d, env_arr, Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = ℝ, Projector_type = :full, conv_info = false, svd_type = :GKL, adjust_χ = false)
    
    sv_arr_old = 0
    S_test_array_old = 0
    S_test_array2_old = 0
    env_arr_old = 0
    number = 10
    #display(loc)
    
    SV_converged = false

    CTMiter = 0
    for i in 1:maxiter_pre
        
        env_arr = CTMRG_step(env_arr, loc, loc_d, Bond_env, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        
        sv_arr = get_corner_sv(env_arr)

        if i>2
            if conv_info
                m = compare_sv(sv_arr, sv_arr_old)
                println("largest SV-difference of the C tensors is $m")
            end
        end

        CTMiter += 1

        #if  i>2 && maximum((S_test_array - S_test_array_old)) < ϵ_pre && maximum(S_test_array2 - S_test_array2_old) < ϵ_pre
        if i>2 && compare_sv(sv_arr, sv_arr_old) isa Number && compare_sv(sv_arr, sv_arr_old) < ϵ_pre
            #number = test_elementwise_convergence(env_arr, env_arr_old, Pattern_arr, 10^-6)
            number = test_elementwise_convergence(env_arr, env_arr_old, 10^-6)

            if SV_converged == false
                if conv_info
                    @info "It took $(i) CTMRG steps to converge the SV to $(ϵ_pre)"
                end
                SV_converged = true
            end

            if number == 0
                if conv_info
                    @info "It took $(i) CTMRG steps to converge the environments element wise to 1e-6"
                end

                #=this is the condition under which we stop our CTMRG-procedure!
                In case we want to dynamically adjust the environment bond dimension we do it here:
                =#
                if adjust_χ != false
                    trunc_sv_arr, env_arr = CTMRG_step(env_arr, loc, loc_d, Bond_env, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_check = true)
            
                    if maximum(trunc_sv_arr) > adjust_χ[1] #if the largest SV cut during generation of the projectors is larger than the threshhold value increase chi
                        while Bond_env < adjust_χ[2] && maximum(trunc_sv_arr) > adjust_χ[1]
                            #@info "the environment bond dimension is being increased from $(Bond_env) to $Bond_env+2"
                            Bond_env = Bond_env + 2
                            trunc_sv_arr, env_arr = CTMRG_step(env_arr, loc, loc_d, Bond_env, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_check = true)
                            test = maximum(trunc_sv_arr)
                            #@info "the largest truncation error is now $(test)"
                        
                        end
                        if Bond_env ≥ adjust_χ[2]
                            @info "the maximal allowed environment bond dimension was reached"
                        end
                    end
                end

                break

            else
                if conv_info
                    println("we are in iteration $i")
                    println("the singular values are converged up to $ϵ_pre but  $number environment tensors are not yet converged element-wise!")
                end
            end

        end

        if i == maxiter_pre
            if compare_sv(sv_arr, sv_arr_old) isa Number && compare_sv(sv_arr, sv_arr_old) < ϵ_pre
                @info "here the singular values did converge but to $(ϵ_pre) the tensors did not converge element wise"
            else
                @info "here the tensors never converged element wise up to 10^-6 and the SV did not converge either"
            end

            #=
            now we should set a limit on the number of CTMRG steps that is quite large. However if the environment tensors do not converge after very many 
            iterations we should not accept this possible PEPS tensor as a result. 
            We can archieve this by just associating a large energy to the PEPS in question. Since we also need to provide a gradient - For this we just choose 
            the shape of the input and fill it with ones... 
            =#
            if adjust_χ != false
                return :unconverged, CTMiter, Bond_env
            end

            return :unconverged, CTMiter

        end

        env_arr_old = env_arr
        sv_arr_old = sv_arr

    end

    if adjust_χ != false
        return env_arr, CTMiter, Bond_env
    end
    
return env_arr, CTMiter
end


function Energy_at_FP(loc, loc_d, env_arr, Bond_env, Pattern_arr; Space_type = ℝ, lattice = :square,  Ham_parameters = nothing, model = :HeisenbergAFM, u_paras = false)
    
    energy_density = calculate_energy_density(env_arr, loc, loc_d, Pattern_arr; Space_type = Space_type, lattice = lattice, Ham_parameters = Ham_parameters, model = model, u_paras = u_paras)

return energy_density
end

function convert_loc_in(loc_in, Pattern_arr; Space_type = ℝ, lattice = :square, identical = false, inputisTM = false)

    if inputisTM 
        PEPS_arr = convert_input(loc_in, Space_type = Space_type, lattice = lattice, identical = identical, inputisTM = true)  
    else
        PEPS_arr = convert_input(loc_in, Space_type = Space_type, lattice = lattice, identical = identical)  
    end
    #Create a dictionary "loc" with pointers from the unit cell positions to the corresponding TensorMaps
    loc = pattern_function(PEPS_arr, Pattern_arr)
    return loc
end

function energy_and_gradient(loc_in, Bond_env, Pattern_arr; maxiter_grad = 100, Space_type = ℝ, maxiter_pre = 500, ϵ_pre = 10^-6, ϵ_grad = 10^-6,
    Projector_type = :half, Ham_parameters = :nothing, model = :Heisenberg_square, lattice = :square, identical = false, sym_tensors = false, D = nothing, Dphys = nothing, 
    reuse_env = (false, nothing), svd_type = :envbond, adjust_χ = false, spiral = false, conv_info = false)

    if spiral == false
        u_paras = false
    else
        u_paras = loc_in[2]
        loc_in = loc_in[1]
    end

    if sym_tensors != false
        loc_in_2 = get_symmetric_tensors(loc_in, D, Dphys, sym_tensors)
        PEPS_arr = convert_input(loc_in_2, Space_type = Space_type, lattice = lattice, identical = identical)  
    else
        loc_in_2 = loc_in

        #if the input of "loc_in" is a Array of TensorMaps, we do no longer need to transform them into TM's!
        if loc_in_2[1] isa TensorMap
            PEPS_arr = convert_input(loc_in_2, Space_type = Space_type, lattice = lattice, identical = identical, inputisTM = true)  
        else
            #here we convert the input into tensor maps and map the original lattice to the square lattice.
            PEPS_arr = convert_input(loc_in_2, Space_type = Space_type, lattice = lattice, identical = identical)  
        end
    
    end
    
    #Here we create an array of pointers, in the shape of the Pattern_arr that points to the relevant tensors in the PEPS_arr
    loc = pattern_function(PEPS_arr, Pattern_arr)

    #Create the adjoint "or Bra" PEPS tensors, and let this be ignores by the AD-engine, such that they behave as independent Tensors
    loc_d = Zygote.@ignore copy(loc)
    
    #=with the keyword reuse_env, we have two variables. The first reuse_env[1] tells us whether we want to reuse the environments at all. If it is taken to be false, 
    we do not reuse environemnts at all and always create them from scratch. If the first variable is true, we only want to reuse the environment under certain conditions, like a small
    gradient in the optimizer etc. . Thus if we do not give a specific environment as the second argument of the keyword "reuse_env" into the function, instead giving "nothing" we still
    default to creating the environments from scratch. Only if we choose the first argument of "reuse_env" to be true and the second is an actual envionment we reuse!
    =#

    if reuse_env[1] == false
        #env_arr = Zygote.@ignore ini_multisite(loc, loc_d, Pattern_arr, PEPS_arr; Space_type = Space_type) 
        env_arr = ini_multisite(PEPS_arr; space_type = Space_type) 

        #converge the environment tensors until all of them are converged element wise
        if adjust_χ == false
            fixed_point_env, CTMiter = converge_environment(loc, loc_d, env_arr, Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, conv_info = conv_info)
        else
            #in case we adjust the environment bond dimension during the creation of the fixed point environments we use the keyword "adjust_χ".
            #in this case we also return the environment bond dimension.
            fixed_point_env, CTMiter, Bond_env = converge_environment(loc, loc_d, env_arr, Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, adjust_χ = adjust_χ)
        end

    else
        if reuse_env[2] == nothing

            env_arr = Zygote.@ignore ini_multisite(loc, loc_d, Pattern_arr, PEPS_arr; Space_type = Space_type) 

            if adjust_χ == false
                fixed_point_env, CTMiter = converge_environment(loc, loc_d, env_arr, Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
            else
                fixed_point_env, CTMiter, Bond_env = converge_environment(loc, loc_d, env_arr, Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, adjust_χ = adjust_χ)
            end
        else 

            if adjust_χ == false
                fixed_point_env, CTMiter = converge_environment(loc, loc_d, reuse_env[2], Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
            else
                fixed_point_env, CTMiter, Bond_env = converge_environment(loc, loc_d, reuse_env[2], Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, adjust_χ = adjust_χ)
            end

            #in a case where we reused the environments and they did not converge, we will try to create them from scratch!
            if fixed_point_env == :unconverged
                display("the environment did not converge when reusing the env's from the previous step")
                env_arr = Zygote.@ignore ini_multisite(loc, loc_d, Pattern_arr, PEPS_arr; Space_type = Space_type) 
                if adjust_χ == false
                    fixed_point_env, CTMiter = converge_environment(loc, loc_d, env_arr, Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
                else
                    fixed_point_env, CTMiter, Bond_env = converge_environment(loc, loc_d, env_arr, Bond_env, Pattern_arr, maxiter_pre, ϵ_pre; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, adjust_χ = adjust_χ)
                end
            end

        end

    end

    if fixed_point_env == :unconverged
    #=if for some reason we encounter a PEPS during a optimization step, for which the CTMRG does not converge element wise,
    we do not want the optimization to stop, but rather just ignore this point and move on.=#
    #MAKE AN EXEPTION FOR THE SPIRAL CASE: in that case the loc_in looks 

        gradient_of_ones = []
        for i in 1:length(loc_in)
            if loc_in[1] isa TensorMap
                #onesTM = ones(ComplexF64, size(loc_in[i].data))
                onesTM = TensorMap(ones, space(loc_in[i]))

                #push!(gradient_of_ones, TensorMap(onesTM, space(loc_in[i])))
                push!(gradient_of_ones, onesTM)
            else
                push!(gradient_of_ones, ones(ComplexF64, size(loc_in[i]))) 
            end
        end


        gradient_of_ones_out = convert(typeof(loc_in), gradient_of_ones)
        if spiral == true
            u_ones = convert(typeof(u_paras), [1.0,1.0])
            gradient_of_ones_out = [gradient_of_ones_out, u_ones]
        end

        #env_arr = Zygote.@ignore ini_multisite(loc, loc_d, Pattern_arr, PEPS_arr; Space_type = Space_type) 
        env_arr = ini_multisite(PEPS_arr; space_type = Space_type) 

        if adjust_χ != false
            return 10.0, gradient_of_ones_out, env_arr, Float64(CTMiter), Bond_env
        end
        
        return 10.0, gradient_of_ones_out, env_arr, Float64(CTMiter)
    end
    

    #Here we calculate the energy and the pullbacks nessessary to build up the gradient at the fixed point of the CTMRG.
    if spiral == false
        energy_density , back_E = Zygote.pullback((x,y) -> Energy_at_FP(loc, x, y, Bond_env, Pattern_arr; Space_type = Space_type, lattice = lattice, Ham_parameters = Ham_parameters, model = model), loc_d, fixed_point_env)
        grad_loc_d = back_E(1)[1];
        grad = back_E(1)[2];
    else
        energy_density , back_E = Zygote.pullback((x,y,z) -> Energy_at_FP(loc, x, y, Bond_env, Pattern_arr; Space_type = Space_type, lattice = lattice, Ham_parameters = Ham_parameters, model = model, u_paras = z), loc_d, fixed_point_env, u_paras)
        grad_loc_d = back_E(1)[1];
        grad = back_E(1)[2];
        grad_u_paras = back_E(1)[3];
        #display(grad_u_paras)
    end

    _, back = Zygote.pullback((x,y) -> CTMRG_step(x, loc, y, Bond_env, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type), fixed_point_env, loc_d); 
    back_env = x -> back(x)[1];
    back_loc_d = x -> back(x)[2];
    
    if loc_in_2[1] isa TensorMap
        back_conv = y -> Zygote.pullback(x -> convert_loc_in(x, Pattern_arr, Space_type = Space_type, lattice = lattice, identical = identical, inputisTM = true), loc_in_2)[2](y)[1]
    else
        back_conv = y -> Zygote.pullback(x -> convert_loc_in(x, Pattern_arr, Space_type = Space_type, lattice = lattice, identical = identical), loc_in_2)[2](y)[1]
    end
    
    result = back_conv(grad_loc_d)
    
    result_old = deepcopy(result)

    #here we build up the gradient until convergence.
    for i in 1:maxiter_grad

        result += back_conv(back_loc_d(grad))

        grad = back_env(grad)

        if norm(result - result_old) < ϵ_grad && i > 7
            break
        end
        
        result_old = result
    end


    gradient_result = result


    #if we used local symmetries we need to make one more pullback.
    if sym_tensors != false
        back_para = y -> Zygote.pullback(x -> get_symmetric_tensors(x, D, Dphys, sym_tensors), loc_in)[2](y)[1]
        gradient_result_sym = back_para(gradient_result)

        if reuse_env[1] == false
            energy_density, gradient_result_sym, nothing, Float64(CTMiter)
        end

        return energy_density, gradient_result_sym, fixed_point_env, Float64(CTMiter)
    end


    if reuse_env[1] == false

        if adjust_χ != false
            
            if spiral == true

                return energy_density, [gradient_result, grad_u_paras], nothing, Float64(CTMiter), Bond_env
            
            end

            return energy_density, gradient_result, nothing, Float64(CTMiter), Bond_env
        end

        if spiral == true

            return energy_density, [gradient_result, grad_u_paras], nothing, Float64(CTMiter)
        
        end
        #display("hello")
        return energy_density, gradient_result, nothing, Float64(CTMiter)
    end


    #if we want to adjust the environment bond dimension during the optimization we have to return the value
    # of the env-bond dim, such that we can use it in the next iteration.
    if adjust_χ != false

        if spiral == true

            return energy_density, [gradient_result, grad_u_paras], fixed_point_env, Float64(CTMiter), Bond_env
        
        end

        return energy_density, gradient_result, fixed_point_env, Float64(CTMiter), Bond_env
    end

    if spiral == true

        return energy_density, [gradient_result, grad_u_paras], fixed_point_env, Float64(CTMiter)
    else

        return energy_density, gradient_result, fixed_point_env, Float64(CTMiter)

    end

end

#multiple dispatch for the energy and gradient function s.th. we give an array of χ. Used for dynamically increasing χ!
function energy_and_gradient(x, χ_arr, Pattern_arr, keywords; adjust_χ = false, reuse_env = false)
    χ = χ_arr[1]

    if adjust_χ == false
        e, gr, fp_env, iter = energy_and_gradient(x, χ, Pattern_arr; maxiter_grad = 100, keywords..., adjust_χ = adjust_χ, reuse_env = reuse_env)
    else
        e, gr, fp_env, iter, χ_adjusted = energy_and_gradient(x, χ, Pattern_arr; maxiter_grad = 100, keywords..., adjust_χ = adjust_χ, reuse_env = reuse_env)

        if χ == χ_adjusted
            @info "we did not need to increase the environement bond dimension, it stays at $(χ)."
        else
            @info "the environment bond dimension was increased in this iteration from $χ to $(χ_adjusted)."
            χ_arr[1] = χ_adjusted
        end
    end
    
    return e, gr, fp_env, iter
end