function apply_loc_unitaries(hopp_term_array, unitary_paras)

    σ_x, σ_y, σ_z, σ_p, σ_m = create_spin_onehalf_operators(; Space_type = ℂ)

    kx = unitary_paras[1]
    ky = unitary_paras[2]

    id = 4 * σ_x * σ_x

    hor_loc_unitary = cos((kx)/2) * id + im*sin((kx)/2) * 2 * σ_y
    vert_loc_unitary = cos((ky)/2) * id + im*sin((ky)/2) * 2 * σ_y
    diag_loc_unitary = cos((kx+ky)/2) * id + im*sin((kx+ky)/2) * 2 * σ_y

    @tensor hor_transf[(i, j);(k, l)] := adjoint(hor_loc_unitary)[j,tbra] * hopp_term_array[1].hor[i,tbra,k,tket] * hor_loc_unitary[tket,l] 
    @tensor vert_transf[(i, j);(k, l)] := adjoint(vert_loc_unitary)[j,tbra] * hopp_term_array[1].vert[i,tbra,k,tket] * vert_loc_unitary[tket,l] 
    @tensor diag_transf[(i, j);(k, l)] := adjoint(diag_loc_unitary)[j,tbra] * hopp_term_array[1].diag[i,tbra,k,tket] * diag_loc_unitary[tket,l] 

    hopp_term_array_transf = [(hor = hor_transf, vert = vert_transf, diag = diag_transf)]

    return hopp_term_array_transf
end

function exp_val_energy_generic_triangular_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = :Heisenberg_square, Space_type = ℂ, loc_terms = true, nn_terms = true, nnn_terms = true, u_paras = false)

    env = pattern_function(env_arr, Pattern_arr)
    
    #= In this part one needs to specify the terms of the Hamiltonian that one wants to include in the expectation value calculation. 
    This can be ignores by the AD-engine. =#
    if model isa Symbol 
        @assert model in keys(model_dict) "The model you specified as a symbol is not in the model_dict - load it with VariPEPSs.model_dict[:keyword] = func"
        ham_term_array = Zygote.@ignore model_dict[model][1](Ham_parameters, loc, Pattern_arr; Space_type = Space_type)
    else
        @assert model isa Array "The functions you import should be passed in an array!"
        ham_term_array = Zygote.@ignore model[1](Ham_parameters, loc, Pattern_arr; Space_type = Space_type)
    end
    
    if u_paras != false
        ham_term_array = apply_loc_unitaries(ham_term_array, u_paras)
    end
    #= now we want to generate all local expectation values! For that we go through all sites in the unit and evaluate the local expectation value. 
    In case there are multiple identical Tensors in the unit cell, we of course only calculate them once =#

    energy_density_local = 0
    
    if haskey(ham_term_array[1], :loc)
        
        for i in minimum(Pattern_arr):maximum(Pattern_arr)
            pos = findfirst(isequal(i), Pattern_arr)

            loc_env = env[pos[1], pos[2]] 
            loc_tensor = loc[pos[1], pos[2]]
            loc_d_tensor = loc_d[pos[1], pos[2]]


            @tensor order = (v2, w1, v1, u1_u, w2_u, l2, y1, y2, x3, u2_d, x2_d, w3, u2_u, x2_u, u1_d, w2_d, l1, x1) begin 
            
                    exp_val_pot_int[] := loc_env.ul[v1,w1] * loc_env.l[v2,w2_d,w2_u,v1] * loc_env.dl[w3,v2]*
                                    loc_env.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1]) *

                                    (ham_term_array[i].loc[l1,l2])*

                                    loc_tensor[w2_u,u2_u,x2_u,u1_u,l2] * loc_env.d[w3,x3,u2_d,u2_u] * 
                                    loc_env.ur[x1,y1]*loc_env.r[x2_d,x2_u,y2,y1]*loc_env.dr[x3,y2] end 

            @tensor order = (w1, v2, v1, y2, y1, x3, x2_u, u2_u, x1, w3, w2_u, u1_u, l1, u1_d, w2_d, u2_d, x2_d) begin 
            
                    norm_pot_int[] := loc_env.ul[v1,w1] * loc_env.l[v2,w2_d,w2_u,v1] * loc_env.dl[w3,v2]*
                                    loc_env.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1]) *
                                    loc_tensor[w2_u,u2_u,x2_u,u1_u,l1] * loc_env.d[w3,x3,u2_d,u2_u] * 
                                    loc_env.ur[x1,y1]*loc_env.r[x2_d,x2_u,y2,y1]*loc_env.dr[x3,y2] end 

            energy_density_local += TensorKit.scalar(exp_val_pot_int) / TensorKit.scalar(norm_pot_int)

        end

    end

    #= now we wish to continue with the next neighbor terms in the Hamiltonian. In the square-lattice geometry we choose the right- and the down-direction
    as the terms we will treat for every site in the unit cell. Again we consider local tensors, that appear multiple times in the unit cell only once. 
    
    For this we need the terms of the Hamiltonian that are connecting the nearest neighbors in horizontal and vertial direction!
    =#
    energy_density_nn = 0
    
    if haskey(ham_term_array[1], :hor)
    
        

        for i in minimum(Pattern_arr):maximum(Pattern_arr)

            #display("this is for site $i")
            pos = findfirst(isequal(i), Pattern_arr)

            #display(pos)

            #if pos = (i,j) we are looking for pos_right = (i+1, j) and pos_down = (i,j+1)! recall we have the notation (x,y)!
            N = size(Pattern_arr)[1]
            M = size(Pattern_arr)[2]
            n = pos[1]
            m = pos[2]
            n_plus = mod(mod((n-1),N) + 1,N) + 1
            m_plus = mod(mod((m-1),M) + 1,M) + 1    


            pos_right = CartesianIndex(n_plus, m)

            pos_down = CartesianIndex(n, m_plus)

            #here we calculate the horizontal expectation value

            env_loc = env[pos[1], pos[2]]
            loc_d_tensor = loc_d[pos[1], pos[2]]
            loc_tensor = loc[pos[1], pos[2]]

            env_loc_r = env[pos_right[1],pos_right[2]]
            loc_d_tensor_r = loc_d[pos_right[1],pos_right[2]]
            loc_tensor_r = loc[pos_right[1],pos_right[2]]

            env_loc_d = env[pos_down[1],pos_down[2]]
            loc_d_tensor_d = loc_d[pos_down[1],pos_down[2]]
            loc_tensor_d = loc[pos_down[1],pos_down[2]]

            @tensor order=(y2,x1,w1,w3, x3,v2, u2b_u,x2_u,w2_d,u2a_d, x2_d,u2b_d,w2_u,u2a_u, u1b_d,u1b_u,y1,v1,u1a_d,u1a_u, e1,e2_d,e2_u,e3, l1,l2,l3,l4) begin 
            
                    val_energy_h[] := env_loc.ul[v1,w1]*env_loc.l[v2,w2_d,w2_u,v1]*env_loc.dl[w3,v2]*
                                        env_loc.u[w1,u1a_d,u1a_u,e1]*conj(loc_d_tensor[w2_d,u2a_d,e2_d,u1a_d,l1])*loc_tensor[w2_u,u2a_u,e2_u,u1a_u,l2]*env_loc.d[w3,e3,u2a_d,u2a_u]*

                                        ham_term_array[i].hor[l1,l3,l2,l4] *

                                        env_loc_r.u[e1,u1b_d,u1b_u,x1]*conj(loc_d_tensor_r[e2_d,u2b_d,x2_d,u1b_d,l3])*loc_tensor_r[e2_u,u2b_u,x2_u,u1b_u,l4]*env_loc_r.d[e3,x3,u2b_d,u2b_u]*
                                        env_loc_r.ur[x1,y1]*env_loc_r.r[x2_d,x2_u,y2,y1]*env_loc_r.dr[x3,y2] end 

            @tensor order=(v1,x3,y1,x1,u1b_d,x2_d,u1b_u,l3,x2_u, u2b_d,u2b_u,y2, e1,e2_d,u1a_d,l1,u1a_u,e2_u, u2a_d,u2a_u,e3, w1,w2_d,w2_u, v2,w3)  begin 
            
                    norm_h[] := env_loc.ul[v1,w1]*env_loc.l[v2,w2_d,w2_u,v1]*env_loc.dl[w3,v2]*
                                        env_loc.u[w1,u1a_d,u1a_u,e1]*conj(loc_d_tensor[w2_d,u2a_d,e2_d,u1a_d,l1])*loc_tensor[w2_u,u2a_u,e2_u,u1a_u,l1]*env_loc.d[w3,e3,u2a_d,u2a_u]*

                                        env_loc_r.u[e1,u1b_d,u1b_u,x1]*conj(loc_d_tensor_r[e2_d,u2b_d,x2_d,u1b_d,l3])*loc_tensor_r[e2_u,u2b_u,x2_u,u1b_u,l3]*env_loc_r.d[e3,x3,u2b_d,u2b_u]*
                                        env_loc_r.ur[x1,y1]*env_loc_r.r[x2_d,x2_u,y2,y1]*env_loc_r.dr[x3,y2] end 
            
            energy_density_nn += TensorKit.scalar(val_energy_h) / TensorKit.scalar(norm_h)

            #here we calculate the vertical expectation value

            @tensor order=(u1,k2,y1,y3, u3,x2, v2_u,u2_u,y2_d,z2_d, u2_d,v2_d,y2_u,z2_u, v1_d,v1_u,k1,x1,z1_d,z1_u, w1,w2_d,w2_u,w3, l1,l2,l3,l4)  begin 
            
                    val_energy_v[] := env_loc.ul[y1,x1] * env_loc.u[x1,y2_d,y2_u,x2] * env_loc.ur[x2,y3] * 
                                        env_loc.l[w1,z1_d,z1_u,y1] * conj(loc_d_tensor[z1_d,w2_d,z2_d,y2_d,l1]) * loc_tensor[z1_u,w2_u,z2_u,y2_u,l2] * env_loc.r[z2_d,z2_u,w3,y3] *

                                        ham_term_array[i].vert[l1,l3,l2,l4] * 

                                        env_loc_d.l[u1,v1_d,v1_u,w1] * conj(loc_d_tensor_d[v1_d,u2_d,v2_d,w2_d,l3]) * loc_tensor_d[v1_u,u2_u,v2_u,w2_u,l4] * env_loc_d.r[v2_d,v2_u,u3,w3] * 
                                        env_loc_d.dl[k1,u1] * env_loc_d.d[k1,k2,u2_d,u2_u] * env_loc_d.dr[k2,u3] end 

            @tensor order=(x1,u3,k1, u1, v1_d,u2_d, v1_u,u2_u,l3,v2_d,v2_u,k2, w1,w2_d,z1_d, z1_u,w2_u,l1, z2_d,z2_u,w3, y1,y2_d,y2_u, x2,y3)  begin 
            
                    norm_v[] := env_loc.ul[y1,x1] * env_loc.u[x1,y2_d,y2_u,x2] * env_loc.ur[x2,y3] * 
                                        env_loc.l[w1,z1_d,z1_u,y1] * conj(loc_d_tensor[z1_d,w2_d,z2_d,y2_d,l1]) * loc_tensor[z1_u,w2_u,z2_u,y2_u,l1] * env_loc.r[z2_d,z2_u,w3,y3] *

                                        env_loc_d.l[u1,v1_d,v1_u,w1] * conj(loc_d_tensor_d[v1_d,u2_d,v2_d,w2_d,l3]) * loc_tensor_d[v1_u,u2_u,v2_u,w2_u,l3] * env_loc_d.r[v2_d,v2_u,u3,w3] * 
                                        env_loc_d.dl[k1,u1] * env_loc_d.d[k1,k2,u2_d,u2_u] * env_loc_d.dr[k2,u3] end 
            
            energy_density_nn += TensorKit.scalar(val_energy_v) / TensorKit.scalar(norm_v)

        end
        
    
    end
    

    energy_density_nnn = 0

    if haskey(ham_term_array[1], :diag)
    
        for i in minimum(Pattern_arr):maximum(Pattern_arr)

            pos = findfirst(isequal(i), Pattern_arr)

            #if pos = (i,j) we are looking for pos_right = (i+1, j) and pos_down = (i,j+1)! recall we have the notation (x,y)!
            N = size(Pattern_arr)[1]
            M = size(Pattern_arr)[2]
            n = pos[1]
            m = pos[2]
            n_plus = mod(mod((n-1),N) + 1,N) + 1
            m_plus = mod(mod((m-1),M) + 1,M) + 1    
            m_minus = mod(m-2,M) + 1

            pos_right = CartesianIndex(n_plus, m)
            pos_down = CartesianIndex(n, m_plus)
            pos_up = CartesianIndex(n, m_minus)
            pos_down_right = CartesianIndex(n_plus, m_plus)
            pos_up_right = CartesianIndex(n_plus, m_minus)


            #here we calculate the horizontal expectation value

            env_loc = env[pos[1], pos[2]]
            loc_d_tensor = loc_d[pos[1], pos[2]]
            loc_tensor = loc[pos[1], pos[2]]

            env_loc_r = env[pos_right[1],pos_right[2]]
            loc_d_tensor_r = loc_d[pos_right[1],pos_right[2]]
            loc_tensor_r = loc[pos_right[1],pos_right[2]]

            env_loc_d = env[pos_down[1],pos_down[2]]
            loc_d_tensor_d = loc_d[pos_down[1],pos_down[2]]
            loc_tensor_d = loc[pos_down[1],pos_down[2]]

            env_loc_u = env[pos_up[1],pos_up[2]]
            loc_d_tensor_u = loc_d[pos_up[1],pos_up[2]]
            loc_tensor_u = loc[pos_up[1],pos_up[2]]

            env_loc_dr = env[pos_down_right[1],pos_down_right[2]]
            loc_d_tensor_dr = loc_d[pos_down_right[1],pos_down_right[2]]
            loc_tensor_dr = loc[pos_down_right[1],pos_down_right[2]]

            env_loc_ur = env[pos_up_right[1],pos_up_right[2]]
            loc_d_tensor_ur = loc_d[pos_up_right[1],pos_up_right[2]]
            loc_tensor_ur = loc[pos_up_right[1],pos_up_right[2]]


            @tensor order = (q1, w3, x3d, p1d, x3u, p1u, r1,     v1, w1, x1d, u1d, x1u, u1u,    w2, x2d, x2u,     v3, z1, u3d, y1d, u3u, y1u, d1,     z3, q3, p3d, y3d, p3u, y3u,     z2, y2d, y2u,     v2, u2u, u2d, p2u, p2d, q2,    l1, l2, dr1, dr2) begin 
            
                    exp_val_hopp_dr[] := env_loc.ul[v1,w1] * env_loc.u[w1,u1d,u1u,w2] * env_loc_r.u[w2,p1d,p1u,w3] * env_loc_r.ur[w3,q1] * 
                                        
                                        env_loc.l[v2,x1d,x1u,v1] * env_loc_r.r[x3d,x3u,q2,q1] * 
                                    
                                        conj(loc_d_tensor[x1d,u2d,x2d,u1d,l2]) * loc_tensor[x1u,u2u,x2u,u1u,l1] * conj(loc_d_tensor_r[x2d,p2d,x3d,p1d,r1]) * loc_tensor_r[x2u,p2u,x3u,p1u,r1]  * 

                                        ham_term_array[i].diag[l2,dr2,l1,dr1] * 
                                    
                                        conj(loc_d_tensor_d[y1d,u3d,y2d,u2d,d1]) * loc_tensor_d[y1u,u3u,y2u,u2u,d1] * conj(loc_d_tensor_dr[y2d,p3d,y3d,p2d,dr2]) * loc_tensor_dr[y2u,p3u,y3u,p2u,dr1] * 
                                    
                                        env_loc_d.l[v3,y1d,y1u,v2] * env_loc_dr.r[y3d,y3u,q3,q2] * 
                                    
                                        env_loc_d.dl[z1,v3] * env_loc_d.d[z1,z2,u3d,u3u] * env_loc_dr.d[z2,z3,p3d,p3u] * env_loc_dr.dr[z3,q3] end 

            @tensor order = (q1, w3, x3d, p1d, x3u, p1u, r1,     v1, w1, x1d, u1d, x1u, u1u, l1,    w2, x2d, x2u,     v3, z1, u3d, y1d, u3u, y1u, d1,     z3, q3, p3d, y3d, p3u, y3u, dr1,     z2, y2d, y2u,     v2, u2u, u2d, p2u, p2d, q2) begin 
            
                    norm_dr[] := env_loc.ul[v1,w1] * env_loc.u[w1,u1d,u1u,w2] * env_loc_r.u[w2,p1d,p1u,w3] * env_loc_r.ur[w3,q1] * 
            
                                        env_loc.l[v2,x1d,x1u,v1] * env_loc_r.r[x3d,x3u,q2,q1] * 
                                    
                                        conj(loc_d_tensor[x1d,u2d,x2d,u1d,l1]) * loc_tensor[x1u,u2u,x2u,u1u,l1] * conj(loc_d_tensor_r[x2d,p2d,x3d,p1d,r1]) * loc_tensor_r[x2u,p2u,x3u,p1u,r1]  * 
                                    
                                        conj(loc_d_tensor_d[y1d,u3d,y2d,u2d,d1]) * loc_tensor_d[y1u,u3u,y2u,u2u,d1] * conj(loc_d_tensor_dr[y2d,p3d,y3d,p2d,dr1]) * loc_tensor_dr[y2u,p3u,y3u,p2u,dr1] * 
                                    
                                        env_loc_d.l[v3,y1d,y1u,v2] * env_loc_dr.r[y3d,y3u,q3,q2] * 
                                    
                                        env_loc_d.dl[z1,v3] * env_loc_d.d[z1,z2,u3d,u3u] * env_loc_dr.d[z2,z3,p3d,p3u] * env_loc_dr.dr[z3,q3] end 

            energy_density_nnn += TensorKit.scalar(exp_val_hopp_dr) / TensorKit.scalar(norm_dr)

        end
    
    end
    
    energy_density_tot_per_site = (energy_density_local + energy_density_nn + energy_density_nnn) / length(minimum(Pattern_arr):maximum(Pattern_arr))

    return real(energy_density_tot_per_site)

end




function exp_val_observables_generic_triangular_lattice_model(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = :Heisenberg_triangular, Space_type = ℂ, loc_terms = true, nn_terms = true)

    env = pattern_function(env_arr, Pattern_arr)
    
    #= In this part one needs to specify the terms of the Hamiltonian that one wants to include in the expectation value calculation. 
    This can be ignores by the AD-engine. =#
    if model isa Symbol 
        @assert model in keys(model_dict) "The model you specified as a symbol is not in the model_dict - load it with VariPEPSs.model_dict[:keyword] = func"
        opp_arr = Zygote.@ignore model_dict[model][2](Ham_parameters, loc, Pattern_arr; Space_type = Space_type)
    else
        @assert model isa Array "The functions you import should be passed in an array!"
        opp_arr = Zygote.@ignore model[2](Ham_parameters, loc, Pattern_arr; Space_type = Space_type)
        #@warn "observables for this model have not been imported to the observables function."
    end
    
    #= now we want to generate all local expectation values! For that we go through all sites in the unit and evaluate the local expectation value. 
    In case there are multiple identical Tensors in the unit cell, we of course only calculate them once =#
    
    #energy_density_local = 0
    
    local_obs_arr = []

    if haskey(opp_arr[1], :loc)
        
        for i in minimum(Pattern_arr):maximum(Pattern_arr)
            pos = findfirst(isequal(i), Pattern_arr)

            loc_env = env[pos[1], pos[2]] 
            loc_tensor = loc[pos[1], pos[2]]
            loc_d_tensor = loc_d[pos[1], pos[2]]

            @tensor order = (w1, v2, v1, y2, y1, x3, x2_u, u2_u, x1, w3, w2_u, u1_u, l1, u1_d, w2_d, u2_d, x2_d) begin 
            
                    norm[] := loc_env.ul[v1,w1] * loc_env.l[v2,w2_d,w2_u,v1] * loc_env.dl[w3,v2]*
                                loc_env.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1]) *
                                loc_tensor[w2_u,u2_u,x2_u,u1_u,l1] * loc_env.d[w3,x3,u2_d,u2_u] * 
                                loc_env.ur[x1,y1]*loc_env.r[x2_d,x2_u,y2,y1]*loc_env.dr[x3,y2] end 

            for j in 1:length(opp_arr[i].loc)

                @tensor order = (v2, w1, v1, u1_u, w2_u, l2, y1, y2, x3, u2_d, x2_d, w3, u2_u, x2_u, u1_d, w2_d, l1, x1) begin 
                
                        exp_val_obs[] := loc_env.ul[v1,w1] * loc_env.l[v2,w2_d,w2_u,v1] * loc_env.dl[w3,v2]*
                                        loc_env.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1]) *

                                        (opp_arr[i].loc[j][l1,l2])*

                                        loc_tensor[w2_u,u2_u,x2_u,u1_u,l2] * loc_env.d[w3,x3,u2_d,u2_u] * 
                                        loc_env.ur[x1,y1]*loc_env.r[x2_d,x2_u,y2,y1]*loc_env.dr[x3,y2] end 
            
                loc_obs_exp_val = TensorKit.scalar(exp_val_obs) / TensorKit.scalar(norm)
                push!(local_obs_arr, loc_obs_exp_val)
            
            end
            
            

        end

    end

    #add nn obervables if needed.
    return local_obs_arr

end
