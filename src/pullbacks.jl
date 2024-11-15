function ChainRulesCore.rrule(::typeof(pinv), S::TensorMap; kwargs...)
    if isdiag(S)
        y = pinv(S; kwargs...)
        function pinv_pullback(ybar)
            fbar = NoTangent()
            
            Δy = unthunk(ybar)
            
            #b = convert_TM_to_mat(y)

            derivTM = TensorMap(zeros, eltype(S), domain(S), domain(S))

            for i in blocks(y)
                Spinv_block = i[2]
                deriv_block = -1 .* diag(Spinv_block).^2
                block(derivTM, i[1]) .= diagm(deriv_block)
            end

            Abar = derivTM * Δy
            
            return fbar, Abar
        end
        return y, pinv_pullback
    else
        error("The funcion pinv() is used and we only defined its pullback for digaonal matrices/tensors! Generalize!")
    end
end

function ChainRulesCore.rrule(::typeof(sqrt), S::TensorMap)
    if isdiag(S)
        y = sqrt(S)
        function sqrt_pullback(ybar)
            fbar = NoTangent()
            
            Δy = unthunk(ybar)
            
            #b = convert_TM_to_mat(y)

            derivTM = TensorMap(zeros, eltype(S), domain(S), domain(S))

            for i in blocks(y)
                ssqrt_block = i[2]
                deriv_block = (1/2) .* (1 ./ diag(ssqrt_block))
                block(derivTM, i[1]) .= diagm(deriv_block)
            end

            Abar = derivTM * Δy
            
            return fbar, Abar
        end
        return y, sqrt_pullback
    else
        error("The funcion sqrt() is used and we only defined its pullback for digaonal matrices/tensors! Generalize!")
    end
end

function ChainRulesCore.rrule(::typeof(sqrt_sv), S::TensorMap)
    if isdiag(S)
        y = sqrt_sv(S)
        function sqrt_sv_pullback(ybar)
            fbar = NoTangent()
            
            Δy = unthunk(ybar)
            
            #b = convert_TM_to_mat(y)

            derivTM = TensorMap(zeros, eltype(S), domain(S), domain(S))

            for i in blocks(y)
                ssqrt_block = i[2]
                deriv_block = (1/2) .* (1 ./ diag(ssqrt_block))
                block(derivTM, i[1]) .= diagm(deriv_block)
            end

            Abar = derivTM * Δy
            
            return fbar, Abar
        end
        return y, sqrt_sv_pullback
    else
        error("The funcion sqrt_sv() is used and we only defined its pullback for digaonal matrices/tensors! Generalize!")
    end
end

function ChainRulesCore.rrule(::typeof(sqrtTM), S::TensorMap)
    y = sqrtTM(S)
    function sqrtTM_pullback(ybar)
        fbar = NoTangent()
        
        Δy = unthunk(ybar)
        
        b = convert_TM_to_mat(y)
        
        deriv = (1/2) .* (1 ./ diag(b))
        
        derivTM = TensorMap(diagm(deriv), codomain(y) ← domain(y))
        
        Abar = derivTM * Δy
        
        return fbar, Abar
    end
    return y, sqrtTM_pullback
end

function ChainRulesCore.rrule(::typeof(p_arr_inv), a::AbstractArray, tol::Number)
    y = p_arr_inv(a, tol)
    function p_arr_inv_pullback(ȳ)
        fbar = NoTangent()
        tolbar = NoTangent()
        
        Δy = unthunk(ȳ)
        
        J = -1 .*  y.^2
        
        abar = J .* Δy 
    
        return fbar, abar, tolbar
    end
    return y, p_arr_inv_pullback
end

function ChainRulesCore.rrule(::typeof(pinv_sqrt), S::TensorMap, tol::Number)
    y = pinv_sqrt(S, tol)   # is that ok? or tol = ...?
    function pinv_sqrt_pullback(ȳ)
        fbar = NoTangent()
        tolbar = NoTangent()
        
        Δy = unthunk(ȳ)
        
        derivative_mat = -1 * (1/2) .* y.data .^3
        
        J = TensorMap(derivative_mat, codomain(y) ← domain(y))
                
        #println("yes i am in fact being used")
        
        abar = J * Δy 
    
        return fbar, abar, tolbar
    end
    return y, pinv_sqrt_pullback
end

#=function ChainRulesCore.rrule(::typeof(TensorKit.scalar), A::TensorMap)
    y = TensorKit.scalar(A)   
    function scalar_pullback(Δy)
        fbar = NoTangent()
        Abar = TensorMap([Δy], codomain(A) ← domain(A))
        return fbar, Abar
    end
    return y, scalar_pullback
end=#

#= function ChainRulesCore.rrule(::typeof(TensorKit.scalar), t::AbstractTensorMap)
    val = scalar(t)
    function scalar_pullback(Δval)
        t2 = similar(t)
        first(blocks(t2))[2][1] = unthunk(Δval)
        return NoTangent(), t2
    end
    return val, scalar_pullback
end =#

function ChainRulesCore.rrule(::typeof(convert_TM_to_mat), A::TensorMap)
    y = convert_TM_to_mat(A)   # is that ok? or tol = ...?
    function convert_TM_to_mat_pullback(ȳ)
        fbar = NoTangent()
        
        Δy = unthunk(ȳ)
        
        Aref = deepcopy(A)
        
        Abar = TensorMap(Δy, codomain(Aref) ← domain(Aref))
        #println("I am being used here")
        return fbar, Abar
    end
    return y, convert_TM_to_mat_pullback
end
#=
function ChainRulesCore.rrule(::typeof(create_Dict_for_TM), s::DataType)
    y = create_Dict_for_TM(s)   
    function create_Dict_for_TM_pullback(ȳ)
        Δy = unthunk(ȳ)
        #display(Δy)
        fbar = NoTangent()
  
        sbar = NoTangent()
        #println("I am being used here, HELOOOO")
        return fbar, sbar
    end
    display(y)
    display(create_Dict_for_TM_pullback)
    return y, create_Dict_for_TM_pullback
end

function ChainRulesCore.rrule(::typeof(convert_Dict_to_TM), A::Dict, codom::VectorSpace, dom::VectorSpace)
    y = convert_Dict_to_TM(A, codom, dom)   
    function convert_Dict_to_TM_pullback(ȳ)
        fbar = NoTangent()
        codombar = NoTangent()
        dombar = NoTangent()

        Δy = unthunk(ȳ)
        
        #Aref = deepcopy(A) #nessessary? 
        
        Abar = convert(Dict,blocks(Δy))
        #println("I am being used here")
        return fbar, Abar, codombar, dombar
    end
    return y, convert_Dict_to_TM_pullback
end

function ChainRulesCore.rrule(::typeof(add_key_to_Dict), D::AbstractDict, key::Sector, val::DenseMatrix)
    y = add_key_to_Dict(D, key, val)   
    function add_key_to_Dict_pullback(ȳ)
        fbar = NoTangent()
        keybar = NoTangent()

        Δy = unthunk(ȳ)
        
        valbar = Δy[key]
        #display(valbar)
        #display(Δy)
        delete!(Δy, key)
        #display(valbar)
        #display(Δy)
        #println("I am being used here")
        return fbar, Δy, keybar, valbar
    end
    return y, add_key_to_Dict_pullback
end
=#
function ChainRulesCore.rrule(::typeof(create_gauge_fixing_tensor), U::TensorMap)
    y = create_gauge_fixing_tensor(U)   
    function create_gauge_fixing_tensor_pullback(ȳ)
        fbar = NoTangent()
        Δy = unthunk(ȳ)

        Ubar = similar(U)

        for i in blocks(Δy) #going through each block of the incoming gradient
            fix_mat, block_pullback_fct = Zygote.pullback(create_block_fix_mat, block(U, i[1])) #calculate the pullback for every block
            Ubar_block = block_pullback_fct(i[2])[1] #the pullback is applied for every block to the incoming gradient
            block(Ubar, i[1]) .= Ubar_block #write the new gradient into the Ubar TensorMap
        end

        return fbar, Ubar
    end
    return y, create_gauge_fixing_tensor_pullback
end


#this is just a minor variation on the tsvd backwards-rule from TensorKitAD
function ChainRulesCore.rrule(::typeof(tsvd_GKL), t::AbstractTensorMap;kwargs...)
    T = eltype(t);

    (U,S,V) = tsvd_GKL(t;kwargs...);

    F = similar(S);
    for (k,dst) in blocks(F)

        src = blocks(S)[k]

        @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
            if abs(src[j,j] - src[i,i])<1e-12
                d = 1e-12
            else
                d = src[j,j]^2-src[i,i]^2
            end

            dst[i,j] = (i == j) ? zero(eltype(S)) : 1/d
        end
    end


    function pullback(v)
        dU,dS,dV = v

        dA = zero(t);
        #A_s bar term
        if dS != ChainRulesCore.ZeroTangent()
            dA += U*_elementwise_mult(dS,one(dS))*V
        end
        #A_uo bar term
        if dU != ChainRulesCore.ZeroTangent()
            J = _elementwise_mult((U'*dU),F)
            dA += U*(J+J')*S*V
        end
        #A_vo bar term
        if dV != ChainRulesCore.ZeroTangent()
            VpdV = V*dV';
            K = _elementwise_mult(VpdV,F)
            dA += U*S*(K+K')*V
        end
        #A_d bar term, only relevant if matrix is complex
        if dV != ChainRulesCore.ZeroTangent() && T <: Complex
            L = _elementwise_mult(VpdV,one(F))
            dA += 1/2*U*pinv(S)*(L' - L)*V
        end

        if codomain(t)!=domain(t)
            pru = U*U';
            prv = V'*V;
            dA += (one(pru)-pru)*dU*pinv(S)*V
            dA += U*pinv(S)*dV*(one(prv)-prv)
        end

        return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
    end
    return (U,S,V), pullback
end