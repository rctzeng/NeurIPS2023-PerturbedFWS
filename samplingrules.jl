using LinearAlgebra
using Polyhedra
import GLPK

"""
Perturbed Frank-Wolfe Sampling
"""
struct PerturbedFWS end
long(sr::PerturbedFWS) = "PerturbedFWS";
abbrev(sr::PerturbedFWS) = "PFWS";
function start(sr::PerturbedFWS, N, t, pep, X0, NA)
    return sr;
end
function nextsample(sr::PerturbedFWS, pep, ★, mca, hμ, hω, t, N, rng)
    K = length(hω);
    if (floor(Int, sqrt(t // K)))^2 == t
        return A = X0[(t % K)+1];
    end
    # Perturbed FW
    n_t = ceil(t^(1/4)); eta = 1/(4 * sqrt(t*K));
    ϵ = 1e-3; θ = 1/(t^(1/4) * exp(sqrt(t)));
    ∇f = [0 for k=1:K];
    for n=1:n_t
	    r = rand(rng); z = randn(rng, K); z = r * z/norm(z); # z ~ Uniform(B)
        _, mca, _ = compute_F(pep.oracle, pep.cstruct, hμ, hω+eta*z, ϵ, θ);
        α = gap(hμ, mca, pep.cstruct) / sum([(★[k] ⊻ mca[k])/(hω[k]+eta*z[k]) for k=1:K]);
        ∇f = ∇f + 0.5 * α^2 * [(★[k] ⊻ mca[k])/(hω[k]^2) for k=1:K];
    end
    return istar(pep.cstruct, ∇f);
end


"""
CombGame (Jourdan et. al.) with OFW and D-Tracking
"""
struct CombGameOFW
end
long(sr::CombGameOFW) = "CombGame-OFW";
abbrev(sr::CombGameOFW) = "CG-OFW";
mutable struct CombGameOFWState
    w;  # vector on simplex
    B; W; NA; # w=sum([W[i]*B[i] for i=1:length(W)])
    w0; ω_OFW_S; ω_OFW_L; diam_A; # for computing gradient of ω-player's loss function
    CombGameOFWState(N, t, X0, NA, diam_A) = new(
        N/t,
        X0, NA/t, NA,
        N/t, 0, zeros(length(N)), diam_A
    )
end
function start(sr::CombGameOFW, N, t, pep, X0, NA)
    As = enumerate_all(pep.cstruct);
    diam_A = maximum([norm(As[i,:]-As[j,:]) for i=1:size(As)[1] for j=1:(i-1)]);
    CombGameOFWState(N, t, X0, NA, diam_A);
end
function nextsample(sr::CombGameOFWState, pep, ★, mca, hμ, hω, t, N, rng)
    K = length(hμ);
    # ω-player
    A = istar(pep.cstruct, -(sr.ω_OFW_S*sr.w - sr.ω_OFW_L));
    # best response λ-player
    ϵ = 1e-3; θ = 1/(t^(1/4) * exp(sqrt(t)));
    _, _, λs, _ = glrt(pep, hμ, sr.w, ϵ, θ);
    # feed optimistic reward to ω-player
    ∇ = optimistic_gradient(pep, hμ, t, N, λs);
    setfield!(sr, :ω_OFW_S, sr.ω_OFW_S + t^(-1/4)/sr.diam_A);
    setfield!(sr, :ω_OFW_L, sr.ω_OFW_L + (t^(-1/4))*sr.w0/sr.diam_A + ∇);
    setfield!(sr, :w, (1-t^(-1/4))*sr.w + (t^(-1/4))*A);
    # sparse D-tracking
    A_idx = findall(x->x==A, sr.B);
    if length(A_idx) == 0
        push!(sr.B, A);
        push!(sr.W, 0);
        push!(sr.NA, 0);
        A_idx = [length(sr.B)];
    end
    e_A = zeros(length(sr.W)); e_A[A_idx[1]] = 1;
    setfield!(sr, :W, (1-t^(-1/4))*sr.W + t^(-1/4)*e_A);
    # next action
    A_idx = argmin(sr.NA ./ sr.W);
    e_A = zeros(length(sr.W)); e_A[A_idx] = 1;
    setfield!(sr, :NA, sr.NA + e_A);
    return sr.B[A_idx];
end
function optimistic_gradient(pep, hμ, t, N, λs)
    [let dist = getexpfam(pep, k),
     ↑ = dup(dist, hμ[k], log(t)/N[k]),
     ↓ = ddn(dist, hμ[k], log(t)/N[k])
     max(d(dist, ↑, λs[k]), d(dist, ↓, λs[k]), log(t)/N[k])
     end
     for k in eachindex(hμ)];
end


"""
Uniform sampling
"""
struct RoundRobin end
long(sr::RoundRobin) = "Uniform";
abbrev(sr::RoundRobin) = "RR";
struct RoundRobinState
    As;
    RoundRobinState(pep) = new(enumerate_all(pep.cstruct));
end
function start(sr::RoundRobin, N, t, pep, X0, NA)
    return RoundRobinState(pep);
end
function nextsample(sr::RoundRobinState, pep, ★, mca, hμ, hω, t, N, rng)
    k = 1+(t % (size(sr.As)[1]));
    return sr.As[k,:];
end
