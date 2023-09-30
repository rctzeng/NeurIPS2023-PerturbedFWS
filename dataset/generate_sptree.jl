######################################################################################################
# This script generates a 5-regular random graph                                                     #
######################################################################################################

using JLD2;
using Graphs
using Random
using SNAPDatasets

include("../cstruct.jl");

seed = 1234;
rng = MersenneTwister(seed);
nV = parse(Int, ARGS[1]); # the number of vertices
r = 5; # degree
@time g_test = random_regular_graph(nV, r, seed=seed); # generates a random 5-regular graph with nV vertices
println("The graph has |V|=$(nv(g_test)) and |E|=$(ne(g_test))");
emap = Dict(e => i for (i,e) in enumerate(edges(g_test)));
sp = SpanningTree(g_test, emap);
println("\t|decls|=$(size(enumerate_all(sp))[1])");
K = ne(sp.graph);
# find μ with unique istar(μ)
μ = zero(K);
Δ = 0;
for n=1:1000
    mu = randn(rng, K) .* 2;
    mu .+= maximum([0, maximum(-mu)]) + 0.1;
    ★1 = istar(sp, mu);
    ★2 = istar(sp, mu, ★1, rng);
    g = dot(★1, mu)-dot(★2, mu);
    if  g > Δ + 1e-10 && g < 0.1
        global μ = mu;
        global Δ = g;
    end
end
println("gap=$Δ");

if Δ > 1e-10
    @save isempty(ARGS) ? "sptree_E$(ne(g_test)).dat" : ARGS[1] sp μ Δ
end
