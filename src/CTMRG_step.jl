function CTMRG_step(env_arr, loc, loc_d, Bond_env, Pattern_arr; Space_type = ℝ, Projector_type = :full, svd_type = :GKL, trunc_check = false)

    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    #perfrom the CTMRG moves in all directions
    
    if trunc_check == true
        trunc_sv_arr = []
    else
        trunc_sv_arr = false
    end
    
    if trunc_sv_arr == false
        env_arr = multi_left_move(env_arr, loc, loc_d, Bond_env, Pattern_arr, Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_arr = trunc_sv_arr)

        env_arr = multi_down_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr, Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_arr = trunc_sv_arr)

        env_arr = multi_right_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr, Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_arr = trunc_sv_arr)

        env_arr = multi_up_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr, Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_arr = trunc_sv_arr)

    else
        env_arr, trunc_sv_arr = multi_left_move(env_arr, loc, loc_d, Bond_env, Pattern_arr, Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_arr = trunc_sv_arr)
        
        env_arr, trunc_sv_arr = multi_down_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr, Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_arr = trunc_sv_arr)

        env_arr, trunc_sv_arr = multi_right_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr, Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_arr = trunc_sv_arr)

        env_arr, trunc_sv_arr = multi_up_move(env_arr, loc, loc_d, Lx, Ly, Bond_env, Pattern_arr, Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_arr = trunc_sv_arr)
        
    end

    if trunc_check == true
        return trunc_sv_arr, env_arr
    else
        return env_arr
    end
end
