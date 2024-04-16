function split_honeycomb(loc, loc_d; Space_type = ℂ)
    buf_loc = Buffer(loc, size(loc))
    buf_loc_d = Buffer(loc_d, size(loc_d))
    
    dim_square = dim(domain(loc[1,1])[3])
    dim_honeycomb = Int(sqrt(dim_square))
    
    split = isomorphism(Space_type^dim_honeycomb ⊗ Space_type^dim_honeycomb, Space_type^dim_square)

    for i in 1:size(loc)[1], j in 1:size(loc)[2]
        
        @tensor loc_split[(i,j);(k,l,m1,m2)] := loc[i,j][i,j,k,l,m] * split[m1, m2, m]
    
        @tensor loc_d_split[(i,j);(k,l,m1,m2)] := loc_d[i,j][i,j,k,l,m] * split[m1, m2, m]
        
        buf_loc[i,j] = loc_split
        
        buf_loc_d[i,j] = loc_d_split

    end

    return copy(buf_loc), copy(buf_loc_d)
end


function exp_val_energy_generic_honeycomb_lattice(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = :KitaevGamma, Space_type = ℂ)
    env = pattern_function(env_arr, Pattern_arr)

    #= create the terms in the hamiltonian that we want to use! Then gives them the labels of the code: 
    loc_terms_A is the sum all one site - terms on site A of the unit cell (analogous for site B), pseuro_loc_term is the term connecting site A and B within the unit
    cell of the honeycomb lattice. Then we further have the horizontal_term and vertical_term. Those describe the terms in the Hamiltonian connecting the unit cell to its 
    neighboring unit cell to the right and to the unit cell below, respectively.=#
    
    if model isa Symbol 
        @assert model in keys(model_dict) "The model you specified as a symbol is not in the model_dict - load it with VariPEPSs.model_dict[:keyword] = func"
        ham_term_array = Zygote.@ignore model_dict[model][1](Ham_parameters, loc, Pattern_arr; Space_type = Space_type)
    else
        @assert model isa Array "The functions you import should be passed in an array!"
        ham_term_array = Zygote.@ignore model[1](Ham_parameters, loc, Pattern_arr; Space_type = Space_type)
    end


    #= In our coarse graining precedure, to map the honeycomb lattice to the square lattice, we combined to adjacent lattice cites onto a single coarse grained site.
    We will now split the physical indices that we combined to one physical leg in the coarse-graining procedure again in two, such that we can calculate expectation values.=#
    
    loc_split, loc_d_split = split_honeycomb(loc, loc_d; Space_type = Space_type)
        
    energy_density_local = 0

    if haskey(ham_term_array[1], :loc)
        
        for i in minimum(Pattern_arr):maximum(Pattern_arr)
            
            pos = findfirst(isequal(i), Pattern_arr)

            env_loc = env[pos[1], pos[2]] 
            loc_tensor = loc_split[pos[1], pos[2]]
            loc_d_tensor = loc_d_split[pos[1], pos[2]]

            @tensor order = (x3, y1, y2, x2_d, u2_d, w1, v2, v1, x1, w3, w2_d, u1_d, w2_u, u1_u, x2_u, u2_u, r1, l1, r2, l2) begin 
            
                    exp_val_pseudo_local[] := env_loc.ul[v1,w1] * env_loc.l[v2,w2_d,w2_u,v1] * env_loc.dl[w3,v2]*
                                            env_loc.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1,r1]) *
                            
                                            (ham_term_array[i].loc[l1, r1, l2, r2]) *
                            
                                            loc_tensor[w2_u,u2_u,x2_u,u1_u,l2,r2] * env_loc.d[w3,x3,u2_d,u2_u] * 
                                            env_loc.ur[x1,y1]*env_loc.r[x2_d,x2_u,y2,y1]*env_loc.dr[x3,y2] end 
                        
            @tensor order = (x3, y1, y2, x2_d, u2_d, w1, v2, v1, x1, w3, w2_d, u1_d, r1, w2_u, u1_u, x2_u, u2_u, l1) begin 
            
            norm_unit_cell[] := env_loc.ul[v1,w1] * env_loc.l[v2,w2_d,w2_u,v1] * env_loc.dl[w3,v2]*
                                env_loc.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1,r1]) *
                
                                loc_tensor[w2_u,u2_u,x2_u,u1_u,l1,r1] * env_loc.d[w3,x3,u2_d,u2_u] * 
                                env_loc.ur[x1,y1]*env_loc.r[x2_d,x2_u,y2,y1]*env_loc.dr[x3,y2] end 

            energy_density_local += TensorKit.scalar(exp_val_pseudo_local) / TensorKit.scalar(norm_unit_cell)

        end
        
    end
    
    energy_density_nn = 0
    
    if haskey(ham_term_array[1], :hor)
    
        for i in minimum(Pattern_arr):maximum(Pattern_arr)

            pos = findfirst(isequal(i), Pattern_arr)

            #if pos = (i,j) we are looking for pos_right = (i+1, j) and pos_down = (i,j+1)! recall we have the notation (x,y)!
            N = size(Pattern_arr)[1]
            M = size(Pattern_arr)[2]
            n = pos[1]
            m = pos[2]
            n_plus = mod(mod((n-1),N) + 1,N) + 1
            m_plus = mod(mod((m-1),M) + 1,M) + 1    

            pos_right = CartesianIndex(n_plus, m)

            pos_down = CartesianIndex(n, m_plus)

            env_loc = env[pos[1], pos[2]] 
            loc_tensor = loc_split[pos[1], pos[2]]
            loc_d_tensor = loc_d_split[pos[1], pos[2]]

            env_loc_r = env[pos_right[1],pos_right[2]]
            loc_d_tensor_r = loc_d_split[pos_right[1],pos_right[2]]
            loc_tensor_r = loc_split[pos_right[1],pos_right[2]]

            env_loc_d = env[pos_down[1],pos_down[2]]
            loc_d_tensor_d = loc_d_split[pos_down[1],pos_down[2]]
            loc_tensor_d = loc_split[pos_down[1],pos_down[2]]
            
            #here we calculate the horizontal expectation value

            @tensor order = (y1, y2, x3, u2b_d, x2_d, u1b_u, r3, x2_u, u2b_u, x1, u1b_d, e3, w1, v2, v1, w2_d, u1a_d, w3, e1, e2_d, u2a_d, l1, w2_u, u1a_u, u2a_u, e2_u, l4, r2, l3, r1) begin 
            
                    exp_val_h[] := env_loc.ul[v1,w1]*env_loc.l[v2,w2_d,w2_u,v1]*env_loc.dl[w3,v2]*
                                    env_loc.u[w1,u1a_d,u1a_u,e1]*conj(loc_d_tensor[w2_d,u2a_d,e2_d,u1a_d,l1,r1])*loc_tensor[w2_u,u2a_u,e2_u,u1a_u,l1,r2]*env_loc.d[w3,e3,u2a_d,u2a_u]*
                
                                    (ham_term_array[i].hor[r1, l3, r2, l4])*
                
                                    env_loc_r.u[e1,u1b_d,u1b_u,x1]*conj(loc_d_tensor_r[e2_d,u2b_d,x2_d,u1b_d,l3,r3])*loc_tensor_r[e2_u,u2b_u,x2_u,u1b_u,l4,r3]*env_loc_r.d[e3,x3,u2b_d,u2b_u]*
                                    env_loc_r.ur[x1,y1]*env_loc_r.r[x2_d,x2_u,y2,y1]*env_loc_r.dr[x3,y2] end 
            
            @tensor order = (w3, v2, u2a_d, w2_d, l1, r1, w2_u, u2a_u, w1, v1, u1a_d, u1a_u, y2, y1, x1, u2b_d, x3, x2_d, x2_u, u1b_d, e3, e2_d, e1, e2_u, u1b_u, u2b_u, l3, r3)  begin 
            
                    norm_h[] := env_loc.ul[v1,w1]*env_loc.l[v2,w2_d,w2_u,v1]*env_loc.dl[w3,v2]*
                                    env_loc.u[w1,u1a_d,u1a_u,e1]*conj(loc_d_tensor[w2_d,u2a_d,e2_d,u1a_d,l1,r1])*loc_tensor[w2_u,u2a_u,e2_u,u1a_u,l1,r1]*env_loc.d[w3,e3,u2a_d,u2a_u]*
                
                                    env_loc_r.u[e1,u1b_d,u1b_u,x1]*conj(loc_d_tensor_r[e2_d,u2b_d,x2_d,u1b_d,l3,r3])*loc_tensor_r[e2_u,u2b_u,x2_u,u1b_u,l3,r3]*env_loc_r.d[e3,x3,u2b_d,u2b_u]*
                                    env_loc_r.ur[x1,y1]*env_loc_r.r[x2_d,x2_u,y2,y1]*env_loc_r.dr[x3,y2] end 
                        
            energy_density_nn += TensorKit.scalar(exp_val_h) / TensorKit.scalar(norm_h)
            
            #here we calculate the vertical expectation value
            
            @tensor order = (y3, x2, y2_d, z2_d, l1, z2_u, y2_u, y1, x1, z1_d, z1_u, k2, u3, v2_u, u2_u, r3, v2_d, u2_d, u1, k1, v1_d, v1_u, w1, w2_u, w2_d, w3, l4, r2, l3, r1) begin 
            
                    exp_val_v[] := env_loc.ul[y1,x1] * env_loc.u[x1,y2_d,y2_u,x2] * env_loc.ur[x2,y3] * 
                                    env_loc.l[w1,z1_d,z1_u,y1] * conj(loc_d_tensor[z1_d,w2_d,z2_d,y2_d,l1,r1]) * loc_tensor[z1_u,w2_u,z2_u,y2_u,l1,r2] * env_loc.r[z2_d,z2_u,w3,y3] *
                
                                    (ham_term_array[i].vert[l3, r1, l4, r2]) *
                
                                    env_loc_d.l[u1,v1_d,v1_u,w1] * conj(loc_d_tensor_d[v1_d,u2_d,v2_d,w2_d,l3,r3]) * loc_tensor_d[v1_u,u2_u,v2_u,w2_u,l4,r3] * env_loc_d.r[v2_d,v2_u,u3,w3] * 
                                    env_loc_d.dl[k1,u1] * env_loc_d.d[k1,k2,u2_d,u2_u] * env_loc_d.dr[k2,u3] end 
            
            
            @tensor order = (y3, x2, z2_d, y2_d, l1, r1, y2_u, z2_u, y1, x1, z1_u, z1_d, w3, k2, u1, k1, v1_d, u2_d, w1, w2_d, u3, v2_d, v1_u, u2_u, v2_u, w2_u, l3, r3) begin 
                    norm_v[] := env_loc.ul[y1,x1] * env_loc.u[x1,y2_d,y2_u,x2] * env_loc.ur[x2,y3] * 
                                    env_loc.l[w1,z1_d,z1_u,y1] * conj(loc_d_tensor[z1_d,w2_d,z2_d,y2_d,l1,r1]) * loc_tensor[z1_u,w2_u,z2_u,y2_u,l1,r1] * env_loc.r[z2_d,z2_u,w3,y3] *
                
                                    env_loc_d.l[u1,v1_d,v1_u,w1] * conj(loc_d_tensor_d[v1_d,u2_d,v2_d,w2_d,l3,r3]) * loc_tensor_d[v1_u,u2_u,v2_u,w2_u,l3,r3] * env_loc_d.r[v2_d,v2_u,u3,w3] * 
                                    env_loc_d.dl[k1,u1] * env_loc_d.d[k1,k2,u2_d,u2_u] * env_loc_d.dr[k2,u3] end 

            energy_density_nn += TensorKit.scalar(exp_val_v) / TensorKit.scalar(norm_v)

            
        end
    
    end
    
    #here we calculate the energy per unit cell in the honeycomb lattice. For the energy per site, devide again by two.
    energy_density = (energy_density_local + energy_density_nn) / length(minimum(Pattern_arr):maximum(Pattern_arr))
                        

return real(energy_density) 
end

function exp_val_observables_generic_honeycomb_lattice(env_arr, loc, loc_d, Ham_parameters, Pattern_arr; model = :KitaevGamma, Space_type = ℂ)
    env = pattern_function(env_arr, Pattern_arr)

    #=
    Here we should load a set of local observables (an array for every square lattice site) that we want to calculate for the model in question. 
    These observables are calculated for both local honeycomb sites corresponding to each square lattice site.
    =#
    if model isa Symbol 
        @assert model in keys(model_dict) "The model you specified as a symbol is not in the model_dict - load it with VariPEPSs.model_dict[:keyword] = func"
        opp_arr = Zygote.@ignore model_dict[model][2](Ham_parameters, loc, Pattern_arr; Space_type = Space_type)
    else
        @assert model isa Array "The functions you import should be passed in an array!"
        opp_arr = Zygote.@ignore model[2](Ham_parameters, loc, Pattern_arr; Space_type = Space_type)
        #@warn "observables for this model have not been imported to the observables function."
    end
    
    #= In our coarse graining precedure, to map the honeycomb lattice to the square lattice, we combined to adjacent lattice cites onto a single coarse grained site.
    We will now split the physical indices that we combined to one physical leg in the coarse-graining procedure again in two, such that we can calculate expectation values.=#
    
    loc_split, loc_d_split = split_honeycomb(loc, loc_d; Space_type = Space_type)
        

    local_obs_arr = []

    if haskey(opp_arr[1], :loc)
        
        for i in minimum(Pattern_arr):maximum(Pattern_arr)
            
            pos = findfirst(isequal(i), Pattern_arr)

            #here we have encountered weirdness with the type of 
            env_loc = env[pos[1], pos[2]] 
            loc_tensor = loc_split[pos[1], pos[2]]
            loc_d_tensor = loc_d_split[pos[1], pos[2]]

            @tensor order = (x3, y1, y2, x2_d, u2_d, w1, v2, v1, x1, w3, w2_d, u1_d, r1, w2_u, u1_u, x2_u, u2_u, l1) begin 
                    
                    norm_unit_cell[] := env_loc.ul[v1,w1] * env_loc.l[v2,w2_d,w2_u,v1] * env_loc.dl[w3,v2]*
                                        env_loc.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1,r1]) *

                                        loc_tensor[w2_u,u2_u,x2_u,u1_u,l1,r1] * env_loc.d[w3,x3,u2_d,u2_u] * 
                                        env_loc.ur[x1,y1]*env_loc.r[x2_d,x2_u,y2,y1]*env_loc.dr[x3,y2] end 

            for j in 1:length(opp_arr[i])

                @tensor order = (x3, y1, y2, x2_d, u2_d, w1, v2, v1, x1, w3, w2_d, u1_d, r1, w2_u, u1_u, x2_u, u2_u, l1, l2) begin 
                
                        exp_loc_A[] := env_loc.ul[v1,w1] * env_loc.l[v2,w2_d,w2_u,v1] * env_loc.dl[w3,v2]*
                                    env_loc.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1,r1]) *
                    
                                    (opp_arr[i].loc[j][l1,l2]) *
                    
                                    loc_tensor[w2_u,u2_u,x2_u,u1_u,l2,r1] * env_loc.d[w3,x3,u2_d,u2_u] * 
                                    env_loc.ur[x1,y1]*env_loc.r[x2_d,x2_u,y2,y1]*env_loc.dr[x3,y2] end 
                    
                loc_obs_exp_val = TensorKit.scalar(exp_loc_A) / TensorKit.scalar(norm_unit_cell)
                push!(local_obs_arr, loc_obs_exp_val)
                #Any[6, Any[7, Any[Any[Any[3, 2], Any[4, 1]], Any[5, Any[Any[9, 10], Any[11, 8]]]]]]
                #Any[6, Any[7, Any[Any[Any[3, 2], Any[4, 1]], Any[5, Any[Any[9, 10], Any[11, 8]]]]]]
                
                @tensor order = (x3, y1, y2, x2_d, u2_d, w1, v2, v1, x1, w3, w2_d, u1_d, l1, w2_u, u1_u, x2_u, u2_u, r1, r2) begin 
                
                        exp_loc_B[] := env_loc.ul[v1,w1] * env_loc.l[v2,w2_d,w2_u,v1] * env_loc.dl[w3,v2]*
                                    env_loc.u[w1,u1_d,u1_u,x1] * conj(loc_d_tensor[w2_d,u2_d,x2_d,u1_d,l1,r1]) *
                    
                                    (opp_arr[i].loc[j][r1,r2]) *
                    
                                    loc_tensor[w2_u,u2_u,x2_u,u1_u,l1,r2] * env_loc.d[w3,x3,u2_d,u2_u] * 
                                    env_loc.ur[x1,y1]*env_loc.r[x2_d,x2_u,y2,y1]*env_loc.dr[x3,y2] end 
                
                loc_obs_exp_val = TensorKit.scalar(exp_loc_B) / TensorKit.scalar(norm_unit_cell)
                push!(local_obs_arr, loc_obs_exp_val)

            end
            

        end
        
    end 
    
    #if nn terms are needed we add them here

return local_obs_arr
end
