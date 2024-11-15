function create_projector_l(env, loc, loc_d, Bond_env, k, l, Lx, Ly; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_out = false)
    m = mod(l-2,Ly)+1
    #if l == 1
    #    m = Ly
    #else
    #    m = l - 1
    #end
    
    n = mod(k,Lx) + 1
    #if k == Lx
    #    n = 1
    #else
    #    n = k+1
    #end
    
    if Projector_type == :half
    
        #options with half of the environemnt:
        @tensor order = (v1, w1, w2u, u1u, a, u1d, w2d) begin Ku[(i,j1,j2);(α,β1,β2)] := env[k,m].ul[w1,v1] * env[k,m].u[v1,w2d,w2u,α] * env[k,m].l[i,u1d,u1u,w1] *
                                                    conj(loc_d[k,m][u1d,j1,β1,w2d,a]) * loc[k,m][u1u,j2,β2,w2u,a] end 
        
        
        
        @tensor order = (w1, v1, w2u, u1u, w2d, u1d, a) begin Kd[(i,j1,j2);(α1,α2,β)] := env[k,l].dl[v1,w1] * env[k,l].d[v1,β,w2d,w2u] * env[k,l].l[w1,u1d,u1u,i] *
                                                    conj(loc_d[k,l][u1d,w2d,α1,j1,a]) * loc[k,l][u1u,w2u,α2,j2,a] end 
    end
    if Projector_type == :full
        
        #options with full of the environemnt:
        @tensor order = (v1, w1, u1d, w2d, u1u, w2u, a, v3, w4, w3d, u3d, u3u, w3u, b, v2, u2u, u2d) begin Ku[(i,j1,j2);(β1,β2,α)] := env[k,m].ul[w1,v1] * env[k,m].u[v1,w2d,w2u,v2] * env[n,m].u[v2,w3d,w3u,v3] * env[n,m].ur[v3,w4] *
                                            env[k,m].l[i,u1d,u1u,w1] * conj(loc_d[k,m][u1d,j1,u2d,w2d,a]) * loc[k,m][u1u,j2,u2u,w2u,a] *
                                            conj(loc_d[n,m][u2d,β1,u3d,w3d,b]) * loc[n,m][u2u,β2,u3u,w3u,b] * env[n,m].r[u3d,u3u,α,w4] end 

        @tensor order = (v3, w4, u3d, w3d, u3u, b, w3u, w1, v1, u1d, w2d, u1u, w2u, a, v2, u2d, u2u) begin Kd[(i,j1,j2);(β1,β2,α)] := env[k,l].dl[v1,w1] * env[k,l].d[v1,v2,w2d,w2u] * env[n,l].d[v2,v3,w3d,w3u] * env[n,l].dr[v3,w4] * 
                                            env[k,l].l[w1,u1d,u1u,i] * conj(loc_d[k,l][u1d,w2d,u2d,j1,a]) * loc[k,l][u1u,w2u,u2u,j2,a] *
                                            conj(loc_d[n,l][u2d,w3d,u3d,β1,b]) * loc[n,l][u2u,w3u,u3u,β2,b] * env[n,l].r[u3d,u3u,w4,α] end 

    end
    
    if Projector_type == :fullfishman
       
        @tensor order = (v1, w1, u1d, w2d, u1u, w2u, a, v3, w4, w3d, u3d, u3u, w3u, b, v2, u2u, u2d) begin Ku_pre[(i,j1,j2);(β1,β2,α)] := env[k,m].ul[w1,v1] * env[k,m].u[v1,w2d,w2u,v2] * env[n,m].u[v2,w3d,w3u,v3] * env[n,m].ur[v3,w4] *
                                            env[k,m].l[i,u1d,u1u,w1] * conj(loc_d[k,m][u1d,j1,u2d,w2d,a]) * loc[k,m][u1u,j2,u2u,w2u,a] *
                                            conj(loc_d[n,m][u2d,β1,u3d,w3d,b]) * loc[n,m][u2u,β2,u3u,w3u,b] * env[n,m].r[u3d,u3u,α,w4] end 

        @tensor order = (v3, w4, u3d, w3d, u3u, b, w3u, w1, v1, u1d, w2d, u1u, w2u, a, v2, u2d, u2u) begin Kd_pre[(i,j1,j2);(β1,β2,α)] := env[k,l].dl[v1,w1] * env[k,l].d[v1,v2,w2d,w2u] * env[n,l].d[v2,v3,w3d,w3u] * env[n,l].dr[v3,w4] * 
                                            env[k,l].l[w1,u1d,u1u,i] * conj(loc_d[k,l][u1d,w2d,u2d,j1,a]) * loc[k,l][u1u,w2u,u2u,j2,a] *
                                            conj(loc_d[n,l][u2d,w3d,u3d,β1,b]) * loc[n,l][u2u,w3u,u3u,β2,b] * env[n,l].r[u3d,u3u,w4,α] end 
        
        #U_u, S_u, V_u_dag = unique_tsvd(Ku_pre, Bond_env, Space_type = Space_type, svd_type = :accuracy)
        U_u, S_u, V_u_dag = unique_tsvd(Ku_pre, χ = Bond_env, svd_type = :full)
        
        #S_u_sqrt = sqrtTM(S_u)
        S_u_sqrt = sqrt(S_u)

        #U_d, S_d, V_d_dag = unique_tsvd(Kd_pre, Bond_env, Space_type = Space_type, svd_type = :accuracy)
        U_d, S_d, V_d_dag = unique_tsvd(Kd_pre, χ = Bond_env, svd_type = :full)
        
        #S_d_sqrt = sqrtTM(S_d)
        S_d_sqrt = sqrt(S_d)

        Ku = U_u * S_u_sqrt
        
        Kd = U_d * S_d_sqrt
        
        Ku = normalization_convention(Ku)
        Kd = normalization_convention(Kd)

        @tensor L[(β);(α)] := Kd[v1,v2,v3,β]*Ku[v1,v2,v3,α]
        
        #U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, Bond_env, Space_type = Space_type, split = :no, svd_type = svd_type);
        U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, χ = Bond_env, svd_type = svd_type, space_type = Space_type);

        #S_L_chi_inv_sqrt = pinv_sqrt(S_L_chi, 10^-8)
        S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-8)

        @tensor Pup[(i,);(j1,j2,j3)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β] * Kd[j1,j2,j3,β]
        @tensor Pdown[(i1,i2,i3);(j,)] := Ku[i1,i2,i3,α] * V_L_chi_d'[α,v2] * S_L_chi_inv_sqrt[v2,j]
        P = (Pu = Pup, Pd = Pdown)

        if trunc_sv_out == true

            #this might need to be ignored for AD - but probably not
            #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
            trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))
            return P, trunc_err 
        else
            return P
        end
        
    end
    
    Ku = normalization_convention(Ku)
    Kd = normalization_convention(Kd)
    
    @tensor L[(β1,β2,β3);(α1,α2,α3)] := Kd[v1,v2,v3,β1,β2,β3]*Ku[v1,v2,v3,α1,α2,α3]

    #display(L)
    #U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, Bond_env, Space_type = Space_type, svd_type = svd_type);
    U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, χ = Bond_env, svd_type = svd_type, space_type = Space_type);
    #display(S_L_chi)
    #S_L_chi_inv_sqrt = pinv_sqrt(S_L_chi, 10^-8)
    S_L_chi_inv_sqrt = pinv(sqrt_sv(S_L_chi); rtol = 10^-8)

    #display(S_L_chi_inv_sqrt)

    @tensor Pup[(i,);(j1,j2,j3)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β1,β2,β3] * Kd[j1,j2,j3,β1,β2,β3]
    @tensor Pdown[(i1,i2,i3);(j,)] := Ku[i1,i2,i3,α1,α2,α3] * V_L_chi_d'[α1,α2,α3,v2] * S_L_chi_inv_sqrt[v2,j]
    
    P = (Pu = Pup, Pd = Pdown)

    if trunc_sv_out == true

        #this might need to be ignored for AD - but probably not
        #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
        #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2))) #take the reals to prevent the value becoming negative at machine precision
        trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2))) #take the reals to prevent the value becoming negative at machine precision

        return P, trunc_err 
    else
        return P
    end
end

function absorb_and_project_tensors_l(env, loc, loc_d, p_dict, k, l, Lx, Ly)
    
    m = mod(l,Ly) + 1

    #if l == Ly
    #    m = 1
    #else
    #    m = l+1
    #end
    
    #keep in mind: p_dict[l] are the projectors above row l & p_dict[l+1] are the ones below. 
    
    @tensor order = (i1,v1,i2,i3) C_ul_tilde[(i,);(j,)] := p_dict[l].Pu[i,i1,i2,i3] * env[k,l].ul[i1,v1] * env[k,l].u[v1,i2,i3,j] 
            
    @tensor order = (i1, v1, i2, i3, v2, a, k1, k2, k3) begin Tr_l_tilde[(i,);(j1,j2,k)] := p_dict[m].Pu[i,i1,i2,i3] * 
                                                env[k,l].l[i1,v1,v2,k1] * conj(loc_d[k,l][v1,i2,j1,k2,a]) * loc[k,l][v2,i3,j2,k3,a] * 
        p_dict[l].Pd[k1,k2,k3,k] end 

    @tensor order = (v1,j1,j2,j3) C_dl_tilde[();(i1,j)] := env[k,l].dl[v1,j1] * env[k,l].d[v1,i1,j2,j3] * p_dict[m].Pd[j1,j2,j3,j] 
    
    #normalize the resulting tensors
    
    C_ul_new = normalization_convention(C_ul_tilde)
    Tr_l_new = normalization_convention(Tr_l_tilde)
    C_dl_new = normalization_convention(C_dl_tilde)
    
    return C_ul_new, Tr_l_new, C_dl_new
end

function update_donealready(arr, k, i, Pattern_arr)
    #this could be done differently, but it gets the job done.
    L = length(arr)
    bf = Buffer(arr, L)
           
    bf[1:L] = arr
    el = Pattern_arr[k,i]
    push!(bf, el)
    return copy(bf)
end

function projectors_l(env, loc, loc_d, Bond_env, k, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(k,Lx)+1

    #if k == Lx
    #    m = 1
    #else
    #    m = k+1
    #end
    
    
    buf = Buffer([], NamedTuple, Ly)
    
    for i in 1:Ly    
        
        if Pattern_arr[m,i] in donealready
            continue
        end
        
        buf[i] = create_projector_l(env, loc, loc_d, Bond_env, k, i, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)

    end
    return copy(buf)
end

#the function below has an additional argument in multiple dispach!
function projectors_l(env, loc, loc_d, Bond_env, k, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    

    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(k,Lx)+1

    #if k == Lx
    #    m = 1
    #else
    #    m = k+1
    #end
    

    proj = Array{NamedTuple}(undef,Ly)
    for i in 1:Ly    
        
        if Pattern_arr[m,i] in donealready
            continue
        end
        
        #as in this version of the function we want to output also the truncations from the projectors we choose the keyword "trunc_sv_out = true"
        proj[i], sv_trunc_ratio = create_projector_l(env, loc, loc_d, Bond_env, k, i, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_out = true)

        append!(trunc_sv_arr, sv_trunc_ratio)

    end


    return proj, trunc_sv_arr
end

function absorb_and_project_l(env, loc, loc_d, p_dict, k, donealready, Pattern_arr)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(k,Lx)+1
    #if k == Lx
    #    m = 1
    #else
    #    m = k+1
    #end
    
    
    
    buf = Buffer([], TensorMap, Ly, 3)
    for i in 1:Ly

        
        if Pattern_arr[m,i] in donealready
            continue
        end
        
        C_ul_new, Tr_l_new, C_dl_new = absorb_and_project_tensors_l(env, loc, loc_d, p_dict, k, i, Lx, Ly)
        buf[i,:] = [C_ul_new, Tr_l_new, C_dl_new]

    end
    return copy(buf)
end

function update_l(new_env, env_arr, k, donealready, Pattern_arr)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(k,Lx)+1
    #if k == Lx
    #    m = 1
    #else
    #    m = k+1
    #end
    
    
    buf = Buffer([], NamedTuple, length(env_arr))
    buf[1:length(env_arr)] = env_arr
    
    for i in 1:length(env_arr)
        
        
        if i in donealready
            continue
        end
        
        
        for (l,j) in enumerate(Pattern_arr[m,:])
        
            if i == j 
                buf[i] = (ul = new_env[l,1], ur = env_arr[i].ur, dl = new_env[l,3], dr = env_arr[i].dr,
                            u = env_arr[i].u, r = env_arr[i].r, d = env_arr[i].d, l = new_env[l,2])

                donealready = Zygote.@ignore update_donealready(donealready, m, l, Pattern_arr)
            end  
        end
    end
    
    return copy(buf), donealready
end

function multi_left_move(env_arr, loc, loc_d, Bond_env, Pattern_arr; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_arr = false)
        
    env = pattern_function(env_arr, Pattern_arr)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    #the absorption happens column by column for all L_x columns of the unit cell
     
    donealready = []
    
    for k in 1:Lx  

        if trunc_sv_arr == false
            p_dict = projectors_l(env, loc, loc_d, Bond_env, k, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        else 
            #here I pass the list of truncations "trunc_sv_arr" as well and use multiple dispach.
            p_dict, trunc_sv_arr = projectors_l(env, loc, loc_d, Bond_env, k, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end
        
        new_env = absorb_and_project_l(env, loc, loc_d, p_dict, k, donealready, Pattern_arr)
        
        #put the updated tensors into the environment array
        #build in a condition for k = Lx
        
        env_arr, donealready = update_l(new_env, env_arr, k, donealready, Pattern_arr)
        
    end

    if trunc_sv_arr == false
        return env_arr
    else
        return env_arr, trunc_sv_arr
    end
    
end


function create_projector_u(env, loc, loc_d, Bond_env, k, l, Lx, Ly; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_out = false)
    
    if k == 1
        m = Lx
    else
        m = k - 1
    end
    if l == Ly
        n = 1
    else
        n = l + 1
    end
    
    if Projector_type == :half

        #options with half of the environemnt:
        @tensor order = (v1, w1, u1u, w2u, a, u1d, w2d) begin Ku[(i,j1,j2);(β1,β2,α)] := env[k,l].ur[w1,v1] * env[k,l].r[w2d,w2u,α,v1] * env[k,l].u[i,u1d,u1u,w1] *
                                                conj(loc_d[k,l][j1,β1,w2d,u1d,a]) * loc[k,l][j2,β2,w2u,u1u,a] end 
            
        @tensor order = (v1, w1, w2u, u1u, w2d, u1d, a) begin Kd[(i,j1,j2);(β,α1,α2)] := env[m,l].ul[v1,w1] * env[m,l].l[β,w2d,w2u,v1] * env[m,l].u[w1,u1d,u1u,i] * 
                                                conj(loc_d[m,l][w2d,α1,j1,u1d,a]) * loc[m,l][w2u,α2,j2,u1u,a] end 
    end
    
    if Projector_type == :full

        #options with full of the environemnt:
        @tensor order = (w4, v3, u3d, w3d, u3u, w3u, b, v1, w1, u1d, w2d, u1u, w2u, a, v2, u2d, u2u) begin Ku[(i,j1,j2);(β1,β2,α)] := env[k,l].ur[w1,v1] * env[k,l].r[w2d,w2u,v2,v1] * env[k,n].r[w3d,w3u,v3,v2] * env[k,n].dr[w4,v3] *
                                            env[k,l].u[i,u1d,u1u,w1] * conj(loc_d[k,l][j1,u2d,w2d,u1d,a]) * loc[k,l][j2,u2u,w2u,u1u,a] *
                                            conj(loc_d[k,n][β1,u3d,w3d,u2d,b]) * loc[k,n][β2,u3u,w3u,u2u,b] * env[k,n].d[α,w4,u3d,u3u] end 



        @tensor order = (w1, v1, w2d, u1d, u1u, a, w2u, v3, w4, w3d, u3d, w3u, u3u, b, v2, u2d, u2u) begin Kd[(i,j1,j2);(β1,β2,α)] := env[m,l].ul[v1,w1] * env[m,l].l[v2,w2d,w2u,v1] * env[m,n].l[v3,w3d,w3u,v2] * env[m,n].dl[w4,v3] *
                                            env[m,l].u[w1,u1d,u1u,i] * conj(loc_d[m,l][w2d,u2d,j1,u1d,a]) * loc[m,l][w2u,u2u,j2,u1u,a] *
                                            conj(loc_d[m,n][w3d,u3d,β1,u2d,b]) * loc[m,n][w3u,u3u,β2,u2u,b] * env[m,n].d[w4,α,u3d,u3u] end 

    end
    
    if Projector_type == :fullfishman
        
        @tensor order = (w4, v3, u3d, w3d, u3u, w3u, b, v1, w1, u1d, w2d, u1u, w2u, a, v2, u2d, u2u) begin Ku_pre[(i,j1,j2);(β1,β2,α)] := env[k,l].ur[w1,v1] * env[k,l].r[w2d,w2u,v2,v1] * env[k,n].r[w3d,w3u,v3,v2] * env[k,n].dr[w4,v3] *
                                            env[k,l].u[i,u1d,u1u,w1] * conj(loc_d[k,l][j1,u2d,w2d,u1d,a]) * loc[k,l][j2,u2u,w2u,u1u,a] *
                                            conj(loc_d[k,n][β1,u3d,w3d,u2d,b]) * loc[k,n][β2,u3u,w3u,u2u,b] * env[k,n].d[α,w4,u3d,u3u] end 



        @tensor order = (w1, v1, w2d, u1d, u1u, a, w2u, v3, w4, w3d, u3d, w3u, u3u, b, v2, u2d, u2u) begin Kd_pre[(i,j1,j2);(β1,β2,α)] := env[m,l].ul[v1,w1] * env[m,l].l[v2,w2d,w2u,v1] * env[m,n].l[v3,w3d,w3u,v2] * env[m,n].dl[w4,v3] *
                                            env[m,l].u[w1,u1d,u1u,i] * conj(loc_d[m,l][w2d,u2d,j1,u1d,a]) * loc[m,l][w2u,u2u,j2,u1u,a] *
                                            conj(loc_d[m,n][w3d,u3d,β1,u2d,b]) * loc[m,n][w3u,u3u,β2,u2u,b] * env[m,n].d[w4,α,u3d,u3u] end 

        
        #U_u, S_u, V_u_dag = unique_tsvd(Ku_pre, Bond_env, Space_type = Space_type, svd_type = :accuracy)
        U_u, S_u, V_u_dag = unique_tsvd(Ku_pre, χ = Bond_env, svd_type = :full)
        
        #S_u_sqrt = sqrtTM(S_u)
        S_u_sqrt = sqrt(S_u)
        
        #U_d, S_d, V_d_dag = unique_tsvd(Kd_pre, Bond_env, Space_type = Space_type, svd_type = :accuracy)
        U_d, S_d, V_d_dag = unique_tsvd(Kd_pre, χ = Bond_env, svd_type = :full)

        #S_d_sqrt = sqrtTM(S_d)
        S_d_sqrt = sqrt(S_d)
        
        Ku = U_u * S_u_sqrt
        
        Kd = U_d * S_d_sqrt
        
        Ku = normalization_convention(Ku)
        Kd = normalization_convention(Kd)

        @tensor L[(β);(α)] := Kd[v1,v2,v3,β]*Ku[v1,v2,v3,α]
        
        #U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, Bond_env, Space_type = Space_type, split = :no, svd_type = svd_type);
        U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, χ = Bond_env, svd_type = svd_type, space_type = Space_type);

        #S_L_chi_inv_sqrt = pinv_sqrt(S_L_chi, 10^-8)
        S_L_chi_inv_sqrt = pinv(sqrt_sv(S_L_chi); rtol = 10^-8)

        @tensor Pup[(i,);(j1,j2,j3)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β] * Kd[j1,j2,j3,β]
        @tensor Pdown[(i1,i2,i3);(j,)] := Ku[i1,i2,i3,α] * V_L_chi_d'[α,v2] * S_L_chi_inv_sqrt[v2,j]

        P = (Pu = Pup, Pd = Pdown)
    
        if trunc_sv_out == true

            #this might need to be ignored for AD - but probably not
            #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
            trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
    
            return P, trunc_err 
        else
            return P
        end
        
    end
    
    Ku = normalization_convention(Ku)
    Kd = normalization_convention(Kd)
    
    @tensor L[(β1,β2,β3);(α1,α2,α3)] := Kd[v1,v2,v3,β1,β2,β3]*Ku[v1,v2,v3,α1,α2,α3]

    #U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, Bond_env, Space_type = Space_type, svd_type= svd_type);
    U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, χ = Bond_env, svd_type= svd_type, space_type = Space_type);


    #S_L_chi_inv_sqrt = pinv_sqrt(S_L_chi, 10^-8)
    S_L_chi_inv_sqrt = pinv(sqrt_sv(S_L_chi); rtol = 10^-8)


    @tensor Pup[(i,);(j1,j2,j3)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β1,β2,β3] * Kd[j1,j2,j3,β1,β2,β3]
    @tensor Pdown[(i1,i2,i3);(j,)] := Ku[i1,i2,i3,α1,α2,α3] * V_L_chi_d'[α1,α2,α3,v2] * S_L_chi_inv_sqrt[v2,j]
    
    P = (Pu = Pup, Pd = Pdown)
    
    if trunc_sv_out == true

        #this might need to be ignored for AD - but probably not
        #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
        #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
        trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))

        return P, trunc_err 
    else
        return P
    end
end

function absorb_and_project_tensors_u(env, loc, loc_d, p_dict, k, l, Lx, Ly)
    
    if k == Lx
        m = 1
    else
        m = k+1
    end
    
    #step1_u: define absorption into the up direction 
    @tensor order = (j1,v1,j2,j3) C_ul_tilde[(i);(j)] := env[k,l].l[i,j2,j3,v1] * env[k,l].ul[v1,j1] * p_dict[k].Pd[j1,j2,j3,j] 

    @tensor order = (i1, i2, v1, i3, v2, a, k1, k2, k3) begin Tr_u_tilde[(i,j1,j2);(k)] := p_dict[k].Pu[i,i1,i2,i3] *
                                                conj(loc_d[k,l][i2,j1,k2,v1,a])*loc[k,l][i3,j2,k3,v2,a]*env[k,l].u[i1,v1,v2,k1] * 
                                                p_dict[m].Pd[k1,k2,k3,k] end 

    @tensor order = (i1, v1, i2, i3) C_ur_tilde[(i,j);()] := p_dict[m].Pu[i,i1,i2,i3] * env[k,l].r[i2,i3,j,v1] * env[k,l].ur[i1,v1] 
            
    C_ul_new = normalization_convention(C_ul_tilde)
    Tr_u_new = normalization_convention(Tr_u_tilde)
    C_ur_new = normalization_convention(C_ur_tilde)

    return C_ul_new, Tr_u_new, C_ur_new
end

function projectors_u(env, loc, loc_d, Bond_env, l, Lx, Ly, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    if l == Ly
        m = 1
    else
        m = l+1
    end
    
    buf = Buffer([], NamedTuple, Lx)
    
    for i in 1:Lx    
        
        if Pattern_arr[i,m] in donealready
            continue
        end

        buf[i] = create_projector_u(env, loc, loc_d, Bond_env, i, l, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)

    end
    return copy(buf)
end

function projectors_u(env, loc, loc_d, Bond_env, l, Lx, Ly, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    if l == Ly
        m = 1
    else
        m = l+1
    end
    
    
    proj = Array{NamedTuple}(undef,Lx)
    for i in 1:Lx    
        
        if Pattern_arr[i,m] in donealready
            continue
        end

        proj[i], sv_trunc_ratio = create_projector_u(env, loc, loc_d, Bond_env, i, l, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_out = true)
        append!(trunc_sv_arr, sv_trunc_ratio)

    end
    return proj, trunc_sv_arr
end

function absorb_and_project_u(env, loc, loc_d, p_dict, l, Lx, Ly, donealready, Pattern_arr)
    if  l == Ly
        m = 1
    else
        m = l+1
    end
    
    buf = Buffer([], TensorMap, 3, Lx)
    for i in 1:Lx

        if Pattern_arr[i,m] in donealready
            continue
        end

        C_ul_new, Tr_u_new, C_ur_new = absorb_and_project_tensors_u(env, loc, loc_d, p_dict, i, l, Lx, Ly)
        buf[:,i] = [C_ul_new, Tr_u_new, C_ur_new]

    end
    return copy(buf)
end

function update_u(new_env, env_arr, l, Lx, Ly, donealready, Pattern_arr)
    if l == Ly
        m = 1
    else
        m = l+1
    end
    
    buf = Buffer([], NamedTuple, length(env_arr))
    buf[1:length(env_arr)] = env_arr
    
    for i in 1:length(env_arr)
        
        if i in donealready
            continue
        end
        
        for (b,j) in enumerate(Pattern_arr[:,m])
        
            if i == j 
                buf[i] = (ul = new_env[1,b], ur = new_env[3,b], dl = env_arr[i].dl, dr = env_arr[i].dr,
                            u = new_env[2,b], r = env_arr[i].r, d = env_arr[i].d, l = env_arr[i].l)


                donealready = @ignore update_donealready(donealready, b, m, Pattern_arr)

            end  
        end
    end
    
    return copy(buf), donealready
end


function multi_up_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_arr = false)
    
    #the absorption happens row by row for all L_y rows of the unit cell

    env = pattern_function(env_arr, Pattern_arr)
    
    donealready = []
    
    for l in 1:Ly  
        
        #create all projectors in this row
        if trunc_sv_arr == false
            p_dict = projectors_u(env, loc, loc_d, Bond_env, l, Lx, Ly, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        else
            p_dict, trunc_sv_arr = projectors_u(env, loc, loc_d, Bond_env, l, Lx, Ly, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end
        #perform absorption and projection for this row
        
        new_env = absorb_and_project_u(env, loc, loc_d, p_dict, l, Lx, Ly, donealready, Pattern_arr)
        
        #put the updated tensors into the environment dictionary
        
        env_arr, donealready = update_u(new_env, env_arr, l, Lx, Ly, donealready, Pattern_arr)
    end
    if trunc_sv_arr == false
        return env_arr
    else
        return env_arr, trunc_sv_arr
    end
end


function create_projector_r(env, loc, loc_d, Bond_env, k, l, Lx, Ly; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_out = false)
    if l == 1
        m = Ly
    else
        m = l - 1
    end
    
    if k == 1
        n = Lx
    else
        n = k-1
    end
    
    if Projector_type == :half

        #options with half of the environment:
        @tensor order = (w4, v3, w3u, u3u, w3d, u3d, b) begin Ku[(α, β1, β2);(k1,k2,l)] := env[k,m].u[α,w3d,w3u,v3] * env[k,m].ur[v3,w4] * conj(loc_d[k,m][β1,k1,u3d,w3d,b]) *
                                                    loc[k,m][β2,k2,u3u,w3u,b] * env[k,m].r[u3d,u3u,l,w4] end
        
        @tensor order = (v3, w4, w3u, u3u, w3d, u3d, b) begin Kd[(α1, α2, β);(k1,k2,l)] := env[k,l].d[β,v3,w3d,w3u] * env[k,l].dr[v3,w4] * conj(loc_d[k,l][α1,w3d,u3d,k1,b]) * 
                                                    loc[k,l][α2,w3u,u3u,k2,b] * env[k,l].r[u3d,u3u,w4,l] end 

    end
    
    if Projector_type == :full

        #options with half of the environment:
        @tensor order = (v1, w1, u1d, w2d, u1u, w2u, a, v3, w4, w3d, u3d, u3u, w3u, b, v2, u2u, u2d) begin Ku[(α, β1, β2);(k1,k2,l)] := env[n,m].ul[w1,v1] * env[n,m].u[v1,w2d,w2u,v2] * env[k,m].u[v2,w3d,w3u,v3] * env[k,m].ur[v3,w4] * 
                                                env[n,m].l[α,u1d,u1u,w1] * conj(loc_d[n,m][u1d,β1,u2d,w2d,a]) * loc[n,m][u1u,β2,u2u,w2u,a] *
                                                conj(loc_d[k,m][u2d,k1,u3d,w3d,b]) * loc[k,m][u2u,k2,u3u,w3u,b] * env[k,m].r[u3d,u3u,l,w4] end 

        @tensor order = (v3, w4, u3d, w3d, u3u, b, w3u, w1, v1, u1d, w2d, u1u, w2u, a, v2, u2d, u2u) begin Kd[(α,β1,β2);(k1,k2,l)] := env[n,l].dl[v1,w1] * env[n,l].d[v1,v2,w2d,w2u] * env[k,l].d[v2,v3,w3d,w3u] * env[k,l].dr[v3,w4] *
                                                env[n,l].l[w1,u1d,u1u,α] * conj(loc_d[n,l][u1d,w2d,u2d,β1,a]) * loc[n,l][u1u,w2u,u2u,β2,a] *
                                                conj(loc_d[k,l][u2d,w3d,u3d,k1,b]) * loc[k,l][u2u,w3u,u3u,k2,b] * env[k,l].r[u3d,u3u,w4,l] end 
    end
    
    if Projector_type == :fullfishman
       
        @tensor order = (v1, w1, u1d, w2d, u1u, w2u, a, v3, w4, w3d, u3d, u3u, w3u, b, v2, u2u, u2d) begin Ku_pre[(α, β1, β2);(k1,k2,l)] := env[n,m].ul[w1,v1] * env[n,m].u[v1,w2d,w2u,v2] * env[k,m].u[v2,w3d,w3u,v3] * env[k,m].ur[v3,w4] * 
                                                env[n,m].l[α,u1d,u1u,w1] * conj(loc_d[n,m][u1d,β1,u2d,w2d,a]) * loc[n,m][u1u,β2,u2u,w2u,a] *
                                                conj(loc_d[k,m][u2d,k1,u3d,w3d,b]) * loc[k,m][u2u,k2,u3u,w3u,b] * env[k,m].r[u3d,u3u,l,w4] end 

        @tensor order = (v3, w4, u3d, w3d, u3u, b, w3u, w1, v1, u1d, w2d, u1u, w2u, a, v2, u2d, u2u) begin Kd_pre[(α,β1,β2);(k1,k2,l)] := env[n,l].dl[v1,w1] * env[n,l].d[v1,v2,w2d,w2u] * env[k,l].d[v2,v3,w3d,w3u] * env[k,l].dr[v3,w4] *
                                                env[n,l].l[w1,u1d,u1u,α] * conj(loc_d[n,l][u1d,w2d,u2d,β1,a]) * loc[n,l][u1u,w2u,u2u,β2,a] *
                                                conj(loc_d[k,l][u2d,w3d,u3d,k1,b]) * loc[k,l][u2u,w3u,u3u,k2,b] * env[k,l].r[u3d,u3u,w4,l] end 
        
        #U_u, S_u, V_u_dag = unique_tsvd(Ku_pre, Bond_env, Space_type = Space_type, svd_type = :accuracy)
        U_u, S_u, V_u_dag = unique_tsvd(Ku_pre, χ = Bond_env, svd_type = :full)
        
        #S_u_sqrt = sqrtTM(S_u)
        S_u_sqrt = sqrt(S_u)
        
        #U_d, S_d, V_d_dag = unique_tsvd(Kd_pre, Bond_env, Space_type = Space_type, svd_type = :accuracy)
        U_d, S_d, V_d_dag = unique_tsvd(Kd_pre, χ = Bond_env, svd_type = :full)
        
        #S_d_sqrt = sqrtTM(S_d)
        S_d_sqrt = sqrt(S_d)
        
        Ku = S_u_sqrt * V_u_dag
        
        Kd = S_d_sqrt * V_d_dag
        
        Ku = normalization_convention(Ku)
        Kd = normalization_convention(Kd)

        @tensor L[(β);(α)] := Kd[β,v1,v2,v3]*Ku[α,v1,v2,v3]
        
        #U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, Bond_env, Space_type = Space_type, split = :no, svd_type = svd_type);
        U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, χ = Bond_env, svd_type = svd_type, space_type = Space_type);


        #S_L_chi_inv_sqrt = pinv_sqrt(S_L_chi, 10^-8)
        S_L_chi_inv_sqrt = pinv(sqrt_sv(S_L_chi); rtol = 10^-8)

        @tensor Pup[(i,);(j1,j2,j3)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β] *Kd[β,j1,j2,j3]
        @tensor Pdown[(i1,i2,i3);(j,)] := Ku[α,i1,i2,i3] * V_L_chi_d'[α,v2] * S_L_chi_inv_sqrt[v2,j]

        P = (Pu = Pup, Pd = Pdown)

        if trunc_sv_out == true

            #this might need to be ignored for AD - but probably not
            #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
            trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
    
            return P, trunc_err 
        else
            return P
        end
        
    end
    
    Ku = normalization_convention(Ku)
    Kd = normalization_convention(Kd)
    
    @tensor L[(β1,β2,β3);(α1,α2,α3)] := Kd[β1,β2,β3,v1,v2,v3]*Ku[α1,α2,α3,v1,v2,v3]

    #U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, Bond_env, Space_type = Space_type, svd_type = svd_type);
    U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, χ = Bond_env, svd_type = svd_type, space_type = Space_type);

    #S_L_chi_inv_sqrt = pinv_sqrt(S_L_chi, 10^-8)
    S_L_chi_inv_sqrt = pinv(sqrt_sv(S_L_chi); rtol = 10^-8)

    
    @tensor Pup[(i,);(j1,j2,j3)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β1,β2,β3] *Kd[β1,β2,β3,j1,j2,j3]
    @tensor Pdown[(i1,i2,i3);(j,)] := Ku[α1,α2,α3,i1,i2,i3] * V_L_chi_d'[α1,α2,α3,v2] * S_L_chi_inv_sqrt[v2,j]
    
    P = (Pu = Pup, Pd = Pdown)

    if trunc_sv_out == true

        #this might need to be ignored for AD - but probably not
        #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
        #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
        trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))


        return P, trunc_err 
    else
        return P
    end
end

function absorb_and_project_tensors_r(env, loc, loc_d, p_dict, k, l, Lx, Ly)
    
    if l == Ly
        m = 1
    else
        m = l+1
    end
        
    #step1_r: define absorption into the right direction
    @tensor order = (j3, v1, j1, j2) C_ur_tilde[(i,j);()] := env[k,l].u[i,j1,j2,v1] * env[k,l].ur[v1,j3] * p_dict[l].Pu[j,j1,j2,j3] 

    @tensor order = (k3, k1, v1, v2, k2, a, j1, j2, j3) begin Tr_r_tilde[(i1,i2,j);(k,)] := p_dict[l].Pd[k1,k2,k3,k] *
                                                conj(loc_d[k,l][i1,j1,v1,k1,a]) * loc[k,l][i2,j2,v2,k2,a] * env[k,l].r[v1,v2,j3,k3] * 
                                                p_dict[m].Pu[j,j1,j2,j3] end 

    @tensor order = (j3, v1, j1, j2) C_dr_tilde[(i,);(j,)] := p_dict[m].Pd[j1,j2,j3,j] * env[k,l].d[i,v1,j1,j2] * env[k,l].dr[v1,j3] 
    
    #normalize the resulting tensors
    C_ur_new = normalization_convention(C_ur_tilde)
    Tr_r_new = normalization_convention(Tr_r_tilde)
    C_dr_new = normalization_convention(C_dr_tilde)

    return C_ur_new, Tr_r_new, C_dr_new
        
end

function projectors_r(env, loc, loc_d, Bond_env, k, Lx, Ly, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    if k == 1
        m = Lx
    else
        m = k-1
    end
    
    buf = Buffer([], NamedTuple, Ly)
    
    for i in 1:Ly    
        
        if Pattern_arr[m,i] in donealready
            continue
        end

        buf[i] = create_projector_r(env, loc, loc_d, Bond_env, k, i, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)

    end
    return copy(buf)
end

function projectors_r(env, loc, loc_d, Bond_env, k, Lx, Ly, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    if k == 1
        m = Lx
    else
        m = k-1
    end
    
    proj = Array{NamedTuple}(undef,Ly)
    
    for i in 1:Ly    
        
        if Pattern_arr[m,i] in donealready
            continue
        end

        proj[i], trunc_sv_ratio = create_projector_r(env, loc, loc_d, Bond_env, k, i, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_out = true)
        append!(trunc_sv_arr, trunc_sv_ratio)
    end
    return proj, trunc_sv_arr
end

function absorb_and_project_r(env, loc, loc_d, p_dict, k, Lx, Ly, donealready, Pattern_arr)
    if k == 1
        m = Lx
    else
        m = k-1
    end
    
    buf = Buffer([], TensorMap, Ly, 3)
    for i in 1:Ly

        if Pattern_arr[m,i] in donealready
            continue
        end

        C_ur_new, Tr_r_new, C_dr_new = absorb_and_project_tensors_r(env, loc, loc_d, p_dict, k, i, Lx, Ly)
        buf[i,:] = [C_ur_new, Tr_r_new, C_dr_new]

    end
    return copy(buf)
end

function update_r(new_env, env_arr, k, Lx, Ly, donealready, Pattern_arr)
    if k == 1
        m = Lx
    else
        m = k-1
    end
    
    buf = Buffer([], NamedTuple, length(env_arr))
    buf[1:length(env_arr)] = env_arr
    
    for i in 1:length(env_arr)
        
        if i in donealready
            continue
        end
        
        for (l,j) in enumerate(Pattern_arr[m,:])
        
            if i == j 
                buf[i] = (ul = env_arr[i].ul, ur = new_env[l,1], dl = env_arr[i].dl, dr = new_env[l,3],
                            u = env_arr[i].u, r = new_env[l,2], d = env_arr[i].d, l = env_arr[i].l)

                donealready = @ignore update_donealready(donealready, m, l, Pattern_arr)
            end  
        end
    end
    
    return copy(buf), donealready
end


function multi_right_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_arr = false)
    
    #the absorption happens row by row for all L_y rows of the unit cell

    env = pattern_function(env_arr, Pattern_arr)
    
    donealready = []
    
    for k in reverse(1:Lx)  
        

        if trunc_sv_arr == false
            p_dict = projectors_r(env, loc, loc_d, Bond_env, k, Lx, Ly, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)

        else
            p_dict, trunc_sv_arr = projectors_r(env, loc, loc_d, Bond_env, k, Lx, Ly, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end
        
        new_env = absorb_and_project_r(env, loc, loc_d, p_dict, k, Lx, Ly, donealready, Pattern_arr)
        
        #put the updated tensors into the environment dictionary
        
        env_arr, donealready = update_r(new_env, env_arr, k, Lx, Ly, donealready, Pattern_arr)
    end
    if trunc_sv_arr == false
        return env_arr
    else
        return env_arr, trunc_sv_arr
    end
    
end



function create_projector_d(env, loc, loc_d, Bond_env, k, l, Lx, Ly; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_out = false)
    if k == 1
        m = Lx
    else
        m = k - 1
    end
    
    if l == 1
        n = Ly
    else
        n = l - 1
    end
    
    if Projector_type == :half

        #options with half of the environment:
        @tensor order = (w4, v3, u3u, w3u, u3d, w3d ,b) begin Ku[(β1, β2, α);(k1,k2,l)] := env[k,l].r[w3d,w3u,v3,α] * env[k,l].dr[w4,v3] * conj(loc_d[k,l][k1,u3d,w3d,β1,b]) * 
                                                    loc[k,l][k2,u3u,w3u,β2,b] * env[k,l].d[l,w4,u3d,u3u] end 
        
        @tensor order = (w4, v3, u3u, w3u, u3d, w3d, b) begin Kd[(β, α1, α2);(k1,k2,l)] := env[m,l].l[v3,w3d,w3u,β] * env[m,l].dl[w4,v3] * conj(loc_d[m,l][w3d,u3d,k1,α1,b]) *
                                                    loc[m,l][w3u,u3u,k2,α2,b] * env[m,l].d[w4,l,u3d,u3u] end 
        
    end
    
    if Projector_type == :full

        #options with the full environment:
        @tensor order = (w4, v3, u3d, w3d, u3u, w3u, b, v1, w1, u1d, w2d, u1u, w2u, a, v2, u2d, u2u) begin Ku[(α, β1, β2);(k1,k2,l)] := env[k,n].ur[w1,v1] * env[k,n].r[w2d,w2u,v2,v1] * env[k,l].r[w3d,w3u,v3,v2] * env[k,l].dr[w4,v3] *
                                        env[k,n].u[α,u1d,u1u,w1] * conj(loc_d[k,n][β1,u2d,w2d,u1d,a]) * loc[k,n][β2,u2u,w2u,u1u,a] *
                                        conj(loc_d[k,l][k1,u3d,w3d,u2d,b]) * loc[k,l][k2,u3u,w3u,u2u,b] * env[k,l].d[l,w4,u3d,u3u] end 

        @tensor order = (w1, v1, w2d, u1d, u1u, a, w2u, v3, w4, w3d, u3d, w3u, u3u, b, v2, u2d, u2u) begin Kd[(α, β1, β2);(k1,k2,l)] := env[m,n].ul[v1,w1] * env[m,n].l[v2,w2d,w2u,v1] * env[m,l].l[v3,w3d,w3u,v2] * env[m,l].dl[w4,v3] * 
                                        env[m,n].u[w1,u1d,u1u,α] * conj(loc_d[m,n][w2d,u2d,β1,u1d,a]) * loc[m,n][w2u,u2u,β2,u1u,a] *
                                        conj(loc_d[m,l][w3d,u3d,k1,u2d,b]) * loc[m,l][w3u,u3u,k2,u2u,b] * env[m,l].d[w4,l,u3d,u3u] end 
        
    end
    
    if Projector_type == :fullfishman
    
        @tensor order = (w4, v3, u3d, w3d, u3u, w3u, b, v1, w1, u1d, w2d, u1u, w2u, a, v2, u2d, u2u) begin Ku_pre[(α, β1, β2);(k1,k2,l)] := env[k,n].ur[w1,v1] * env[k,n].r[w2d,w2u,v2,v1] * env[k,l].r[w3d,w3u,v3,v2] * env[k,l].dr[w4,v3] *
                                        env[k,n].u[α,u1d,u1u,w1] * conj(loc_d[k,n][β1,u2d,w2d,u1d,a]) * loc[k,n][β2,u2u,w2u,u1u,a] *
                                        conj(loc_d[k,l][k1,u3d,w3d,u2d,b]) * loc[k,l][k2,u3u,w3u,u2u,b] * env[k,l].d[l,w4,u3d,u3u] end 

        @tensor order = (w1, v1, w2d, u1d, u1u, a, w2u, v3, w4, w3d, u3d, w3u, u3u, b, v2, u2d, u2u) begin Kd_pre[(α, β1, β2);(k1,k2,l)] := env[m,n].ul[v1,w1] * env[m,n].l[v2,w2d,w2u,v1] * env[m,l].l[v3,w3d,w3u,v2] * env[m,l].dl[w4,v3] * 
                                        env[m,n].u[w1,u1d,u1u,α] * conj(loc_d[m,n][w2d,u2d,β1,u1d,a]) * loc[m,n][w2u,u2u,β2,u1u,a] *
                                        conj(loc_d[m,l][w3d,u3d,k1,u2d,b]) * loc[m,l][w3u,u3u,k2,u2u,b] * env[m,l].d[w4,l,u3d,u3u] end 
        
        #U_u, S_u, V_u_dag = unique_tsvd(Ku_pre, Bond_env, Space_type = Space_type, svd_type = :accuracy)
        U_u, S_u, V_u_dag = unique_tsvd(Ku_pre, χ = Bond_env, svd_type = :full)
        
        #S_u_sqrt = sqrtTM(S_u)
        S_u_sqrt = sqrt(S_u)

        #U_d, S_d, V_d_dag = unique_tsvd(Kd_pre, Bond_env, Space_type = Space_type, svd_type = :accuracy)
        U_d, S_d, V_d_dag = unique_tsvd(Kd_pre, χ = Bond_env, svd_type = :full)
        
        #S_d_sqrt = sqrtTM(S_d)
        S_d_sqrt = sqrt(S_d)

        Ku = S_u_sqrt * V_u_dag
        
        Kd = S_d_sqrt * V_d_dag
        
        Ku = normalization_convention(Ku)
        Kd = normalization_convention(Kd)

        @tensor L[(β);(α)] := Kd[β,v1,v2,v3]*Ku[α,v1,v2,v3]
        
        #U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, Bond_env, Space_type = Space_type, split = :no, svd_type = svd_type);
        U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, χ = Bond_env, svd_type = svd_type, space_type = Space_type);


        #S_L_chi_inv_sqrt = pinv_sqrt(S_L_chi, 10^-8)
        S_L_chi_inv_sqrt = pinv(sqrt_sv(S_L_chi); rtol = 10^-8)

        @tensor Pup[(i,);(j1,j2,j3)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β] *Kd[β,j1,j2,j3]
        @tensor Pdown[(i1,i2,i3);(j,)] := Ku[α,i1,i2,i3] * V_L_chi_d'[α,v2] * S_L_chi_inv_sqrt[v2,j]

        P = (Pu = Pup, Pd = Pdown)

        if trunc_sv_out == true

            #this might need to be ignored for AD - but probably not
            #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
            trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
    
            return P, trunc_err 
        else
            return P
        end
        
    end
    
    Ku = normalization_convention(Ku)
    Kd = normalization_convention(Kd)
    
    @tensor L[(β1,β2,β3);(α1,α2,α3)] := Kd[β1,β2,β3,v1,v2,v3] * Ku[α1,α2,α3,v1,v2,v3]

    #U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, Bond_env, Space_type = Space_type, svd_type = svd_type);
    U_L_chi, S_L_chi, V_L_chi_d = unique_tsvd(L, χ = Bond_env, svd_type = svd_type, space_type = Space_type);

    #S_L_chi_inv_sqrt = pinv_sqrt(S_L_chi, 10^-8)
    S_L_chi_inv_sqrt = pinv(sqrt_sv(S_L_chi); rtol = 10^-8)

    @tensor Pup[(i,);(j1,j2,j3)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β1,β2,β3] *Kd[β1,β2,β3,j1,j2,j3]
    @tensor Pdown[(i1,i2,i3);(j,)] := Ku[α1,α2,α3,i1,i2,i3] * V_L_chi_d'[α1,α2,α3,v2] * S_L_chi_inv_sqrt[v2,j]
    #display(Pd)
    
    P = (Pu = Pup, Pd = Pdown)
    
    if trunc_sv_out == true

        #this might need to be ignored for AD - but probably not
        #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
        #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
        trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))

        return P, trunc_err 
    else
        return P
    end
end

function absorb_and_project_tensors_d(env, loc, loc_d, p_dict, k, l, Lx, Ly)
    
    if k == Lx
        m = 1
    else
        m = k + 1
    end
    
    #step1_d: absorption
    @tensor order = (i3, v1, i1, i2) C_dl_tilde[();(i,j)] := env[k,l].dl[i3,v1] * env[k,l].l[v1,i1,i2,j] * p_dict[k].Pd[i1,i2,i3,i] 

    @tensor order = (i3, i1, v1, i2, v2, a, j1, j2, j3) begin Tr_d_tilde[(i);(j,k1,k2)] := p_dict[k].Pu[i,i1,i2,i3] * 
                                                    env[k,l].d[i3,j3,v1,v2] * conj(loc_d[k,l][i1,v1,j1,k1,a]) * loc[k,l][i2,v2,j2,k2,a] *
        p_dict[m].Pd[j1,j2,j3,j] end 

    @tensor order = (i3, v1, i1, i2) C_dr_tilde[(i,);(j,)] := p_dict[m].Pu[i,i1,i2,i3] * env[k,l].dr[i3,v1] * env[k,l].r[i1,i2,v1,j] 
            
    #normalize the resulting tensors
    C_dl_new = normalization_convention(C_dl_tilde)
    Tr_d_new = normalization_convention(Tr_d_tilde)
    C_dr_new = normalization_convention(C_dr_tilde)

    return C_dl_new, Tr_d_new, C_dr_new
end

function projectors_d(env, loc, loc_d, Bond_env, l, Lx, Ly, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    if l == 1
        m = Ly
    else
        m = l-1
    end
    
    buf = Buffer([], NamedTuple, Lx)
    
    for i in 1:Lx    
        
        if Pattern_arr[i,m] in donealready
            continue
        end

        buf[i] = create_projector_d(env, loc, loc_d, Bond_env, i, l, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)

    end
    return copy(buf)
end

function projectors_d(env, loc, loc_d, Bond_env, l, Lx, Ly, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    if l == 1
        m = Ly
    else
        m = l-1
    end
    
    proj = Array{NamedTuple}(undef,Lx)
    
    for i in 1:Lx    
        
        if Pattern_arr[i,m] in donealready
            continue
        end

        proj[i], sv_trunc_ratio = create_projector_d(env, loc, loc_d, Bond_env, i, l, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_out = true)
        append!(trunc_sv_arr, sv_trunc_ratio)
    end
    return proj, trunc_sv_arr
end

function absorb_and_project_d(env, loc, loc_d, p_dict, l, Lx, Ly, donealready, Pattern_arr)
    if  l == 1
        m = Ly
    else
        m = l-1
    end
    
    buf = Buffer([], TensorMap, 3, Lx)
    for i in 1:Lx

        if Pattern_arr[i,m] in donealready
            continue
        end

        C_dl_new, Tr_d_new, C_dr_new = absorb_and_project_tensors_d(env, loc, loc_d, p_dict, i, l, Lx, Ly)
        buf[:,i] = [C_dl_new, Tr_d_new, C_dr_new]

    end
    return copy(buf)
end

function update_d(new_env, env_arr, l, Lx, Ly, donealready, Pattern_arr)
    if l == 1
        m = Ly
    else
        m = l-1
    end
    
    buf = Buffer([], NamedTuple, length(env_arr))
    buf[1:length(env_arr)] = env_arr
    
    for i in 1:length(env_arr)
        
        if i in donealready
            continue
        end
        
        for (b,j) in enumerate(Pattern_arr[:,m])
        
            if i == j 
                buf[i] = (ul = env_arr[i].ul, ur = env_arr[i].ur, dl = new_env[1,b], dr = new_env[3,b],
                            u = env_arr[i].u, r = env_arr[i].r, d = new_env[2,b], l = env_arr[i].l)


                donealready = @ignore update_donealready(donealready, b, m, Pattern_arr)

            end  
        end
    end
    
    return copy(buf), donealready
end

function multi_down_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_arr = false)
    
    #the absorption happens row by row for all L_y rows of the unit cell

    env = pattern_function(env_arr, Pattern_arr)
    
    donealready = []
    
    for l in reverse(1:Ly)   

        #create projectors for the row l
        if trunc_sv_arr == false
            p_dict = projectors_d(env, loc, loc_d, Bond_env, l, Lx, Ly, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        else
            p_dict, trunc_sv_arr = projectors_d(env, loc, loc_d, Bond_env, l, Lx, Ly, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end
        #absorb and project the tensors in the row l

        new_env = absorb_and_project_d(env, loc, loc_d, p_dict, l, Lx, Ly, donealready, Pattern_arr)
        
        #put the updated tensors into the environment dictionary
       
        env_arr, donealready = update_d(new_env, env_arr, l, Lx, Ly, donealready, Pattern_arr)
    end
    if trunc_sv_arr == false
        return env_arr
    else
        return env_arr, trunc_sv_arr
    end
end