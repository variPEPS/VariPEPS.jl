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

function ChainRulesCore.rrule(::typeof(TensorKit.scalar), A::TensorMap)
    y = TensorKit.scalar(A)   
    function scalar_pullback(Δy)
        fbar = NoTangent()
        Abar = TensorMap([Δy], codomain(A) ← domain(A))
        return fbar, Abar
    end
    return y, scalar_pullback
end

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