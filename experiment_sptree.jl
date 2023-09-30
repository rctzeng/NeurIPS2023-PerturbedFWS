using JLD2;
using Distributed;
using Printf;
@everywhere include("runit.jl");
@everywhere include("utilities/thresholds.jl");
include("utilities/experiment_helpers.jl");

dist = Gaussian();
seed = 1234;
δs = (0.1,);
βs = GK16.(δs);
N = 100;
rng = MersenneTwister(seed);
##############################################################
K = parse(Int, ARGS[1]);
@load "dataset/sptree_E$K.dat" sp μ Δ
println("The graph has |V|=$(nv(sp.graph)), |E|=$K and Δ_min(μ)=$Δ");
##############################################################
MAX_ITER = parse(Int, ARGS[2]);
#mcp = NaiveMCP();
mcp = NoRegMCP(MAX_ITER, rng);
pep = CombBestAction(dist, sp, mcp);
# sampling rules to be compared
srs = [
    PerturbedFWS(),
    CombGameOFW(),
    RoundRobin(),
];
println("μ=$μ, N=$N");
# compute
@time data = pmap(
    ((sr,i),) -> runit(seed+i, sr, μ, pep, δs, βs),
    Iterators.product(srs, 1:N)
);

dump_stats(pep, μ, δs, βs, srs, data, N);
# save
@save "BAI_sptree_E$K.dat" dist μ pep srs data δs βs N seed

# visualise by loading viz_bai.jl
