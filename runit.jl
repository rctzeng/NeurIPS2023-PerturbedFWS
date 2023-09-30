using Random;
using CPUTime;
include("utilities/expfam.jl");
include("mcp_oracle.jl");
include("samplingrules.jl");

# Run the learning algorithm, paramterised by a sampling rule
# The stopping and recommendation rules are common
# βs must be a list of thresholds *in increasing order*

function play!(A, rng, pep, μs, N, S, K)
    Y = [sample(rng, getexpfam(pep, k), μs[k]) for k=1:K];
    S .+= A .* Y;
    N .+= A;
end

function runit(seed, sr, μs, pep, δs, βs)
    βs = collect(βs); # mutable copy
    rng = MersenneTwister(seed);
    K = length(μs);
    N = zeros(Int64, K); t = 0;     # counts on base arms
    S = zeros(K);                   # sum of samples
    baseline = CPUtime_us();

    # covering initialization: ensuring each arm is pull at least once
    X0 = []; W = [];
    for k in 1:K
        e_k = zeros(K); e_k[k] = 1;
        A = istar(pep.cstruct, e_k);
        push!(X0, A);
        n = 0;
        while N[k] == 0
            play!(A, rng, pep, μs, N, S, K); t += 1; n+=1;
        end
        push!(W, n);
    end

    state = start(sr, N, t, pep, X0, W);
    R = Tuple{Array{Int64,1}, UInt64, Array{Int64,1}, UInt64}[]; # collect return values
    while true
        hμ = S./N; # emp. estimates
        hω = N/sum(N);
        # test stopping criterion
        ★, mca, _, F = glrt(pep, hμ, hω, 1e-3, δs[1]/(t^2));
        while t * F > (1+1e-3) * βs[1](t) && (dot(★, hμ)-dot(istar(pep.cstruct, hμ, ★, rng),hμ))>1e-7
            popfirst!(βs);
            push!(R, (collect(★), t, copy(N), CPUtime_us()-baseline));
            if isempty(βs)
                println("\t\t[$(long(sr))] τ_δ=$t, correctness=$(dot(μs,istar(pep, µs))==dot(μs,★))");
                return R;
            end
        end
        # invoke sampling rule
        A = nextsample(state, pep, ★, mca, hμ, hω, t, N, rng);
        # sample
        play!(A, rng, pep, μs, N, S, K); t += 1;
        if t % 1000 == 0
            println("[t=$t] tF=$(t*F), β(t)=$(βs[1](t))");
        end
    end
end
