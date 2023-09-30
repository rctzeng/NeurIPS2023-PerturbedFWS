######################################################################################################
# A combinatorial best-action identification (CombBestAction) problem is parameterized by            #
#  - expfam: the reward distributions                                                                #
#  - μ: the expected reward of K arms, stored in an array                                            #
#  - the answer set is specified via `cstruct`, i.e., the underlying of combinatorial structure      #
# A CombBestAction provides the following functions:                                                 #
#  - istar: correct answer for feasible μ                                                            #
#  - glrt: value and best answer to (μ, ω)                                                           #
######################################################################################################
using Random

include("cstruct.jl")

struct CombBestAction
    expfam; # the class of reward distributions
    cstruct; # the underlying combinatorial structure of the answer set
    oracle; # the oracle for computing the most confusing action and F_μ(ω)
end
istar(pep::CombBestAction, μ) = istar(pep.cstruct, μ);
getexpfam(pep::CombBestAction, k) = pep.expfam;

function glrt(pep::CombBestAction, μ, ω, ϵ, θ)
    K = length(ω);
    ★, a_star, F = compute_F(pep.oracle, pep.cstruct, μ, ω, ϵ, θ) # ★, most confusing action, F_μ(ω)
    α = dot(★ - a_star, μ) / sum([abs(★[k]-a_star[k])/ω[k] for k=1:K]);
    λ = [μ[k] + α * (a_star[k] - ★[k])/ω[k] for k=1:K];
    return ★, a_star, λ, F;
end
function gap(μ, a, cst)
    ★ = istar(cst, μ);
    return dot((★ - a), μ);
end

######################################################################################################
# Algorithms for computing the most confusing parameter (MCP)                                        #
######################################################################################################
"""
NoRegMCP implements Algorithm 1 `(ε,θ)-MCP(ω,μ)`
 - compute_F: when MAX_ITER = ∞, the output enjoys is guaranteed to be a (1+ϵ)-approx to F_μ(ω) w.p. >= 1-θ
"""
struct NoRegMCP
    MAX_ITER; # max iterations of the no-regret algorithm
    rng; # the pseuorandom generator
    NoRegMCP(MAX_ITER, rng) = new(MAX_ITER, rng);
end
function compute_F(orl::NoRegMCP, cst, μ, ω, ϵ, θ)
    K = length(μ); D = max_one_norm(cst);
    μ_max = maximum(μ); ω_1 = sum([1/ω[k] for k=1:K if ω[k]>1E-10]);
    ★ = istar(cst, μ);
    L = 4 * D^2 * K * μ_max^2 * ω_1;
    c = L * (4 * sqrt(K * (log(K)+1)) + sqrt(log(1.0/θ)/2));
    lsum = zeros(K);
    g_best, A_best, alp_best, s = typemax(Float64), similar(★), -1, 1;
    η1 = sqrt(K*(log(K)+1) / (2*L^2))
    while ((s==1) || (s <= (c*(1+ϵ)/(ϵ*g_best))^2)) && (s < orl.MAX_ITER)
        Z = randexp(orl.rng, K);
        A = istar(cst, -((η1 / sqrt(s)) .* lsum + Z), ★, orl.rng); # action-player plays Follow-The-Perturbed-Leader
        α = _alpha_best(★, A, μ, ω); # α-player plays Best-Response
        lsum += _l(α, μ, ω, ★);
        cur_g = _g(A, α, μ, ω, ★);
        if cur_g < g_best
            g_best = cur_g; A_best = A; alp_best = α;
        end
        s += 1;
    end
    @assert A_best != ★ "the most confusing superarm cannot be ★"
    return ★, A_best, _g(A_best, alp_best, μ, ω, ★); # the most confusing action, F_μ(ω)
end
function _alpha_best(★, A, μ, ω)
    return sum([μ[k]*(★[k]-A[k]) for k=1:length(ω)]) / sum([(★[k] ⊻ A[k])/ω[k] for k=1:length(ω)]);
end
function _g(A, α, μ, ω, ★)
    return _c(α, μ, ω, ★) + dot(_l(α, μ, ω, ★), A);
end
function _c(α, μ, ω, ★)
    return α * sum([(-0.5 * α /ω[k] + μ[k]) * ★[k] for k=1:length(ω)]);
end
function _l(α, μ, ω, ★)
    return [-(α * μ[k] + α^2/(2*ω[k]) * (1 - 2 * ★[k])) for k=1:length(ω)];
end

"""
Naive implementation of MCP oracle
"""
struct NaiveMCP
end
function compute_F(orl::NaiveMCP, cst, μ, ω, ϵ, θ)
    """ the closed-form is for Gaussian with unit variance """
    decls = enumerate_all(cst); M = size(decls)[1];
    ★ = istar(cst, μ);
    # compute the numerator
    gaps = zeros(M);
    sub_istar_sup = transpose(★) .- decls;
    mul!(gaps, sub_istar_sup, μ);
    # compute the denominator
    de = zeros(M);
    mul!(de, (transpose(★) .- decls) .^ 2, 1.0 ./ ω);
    # compute f for each superarms in decision class
    istar_idx = collect(symdiff(Set(rowvals(sub_istar_sup)), Set(1:M)))[1];
    fs = [(i != istar_idx) ? (gaps[i]^2 / (2 * de[i])) : typemax(Float64) for i=1:M]
    min_idx = argmin(fs);
    A_best, g_best = decls[min_idx, :], fs[min_idx];
    @assert A_best != ★ "the most confusing superarm must not be ★"
    return ★, A_best, g_best; # the most confusing superarm, F_μ(ω)
end
