######################################################################################################
# Functions for a combinatoral structure A include:                                                  #
#  - istar(cst::A, μ): output any action in $\argmax_{x ∈ X}⟨μ, x⟩$                                  #
#  - istar(cst::A, μ, ★, rng): output any action in $\argmax_{x ∈ X\{★}}⟨μ, x⟩$                      #
#  - max_one_norm(cst::A): the maximum number of arms contained in any action x ∈ X                  #
#  - enumerate_all(cst::A): output all actions in the combinatorials set X
######################################################################################################

using Graphs # for minimum spanning tree algorithms
using Combinatorics # for generating the combinations
using LinearAlgebra # for computing determinant
using SparseArrays # for accessing a submatrix of a sparse matrix

include("graph-algos/sp_prim_mst.jl")
include("cstruct_helper.jl")

""" Spanning Tree """
struct SpanningTree
    graph::SimpleGraph; # unweighted graph
    emap::Dict{Edge,Int}; # maps an edge to an index for accessing μ
    SpanningTree(g, e_map) = new(g, e_map);
end
function max_one_norm(cst::SpanningTree) # corresponds to $D$ in the paper
    return nv(cst.graph)-1;
end
function istar(cst::SpanningTree, μ) # corresponds to argmax_{x ∈ X}⟨μ, x⟩ in the paper
    """ find a max-weight spanning tree in O(|E|ln|V|) time """
    ★, _ = _istar(cst, μ);
    return ★; # return a sparse vector
end
function _istar(cst::SpanningTree, μ)
    """ returns a max-weight spanning tree and its edges"""
    st_edges = sp_prim_mst(cst.graph, -_get_edge_weights(cst.graph, cst.emap, μ));
    ★ = sparsevec([_lookup_edge(cst.emap, e) for e in st_edges], fill(one(Int), length(st_edges)), length(μ));
    @assert sum(★) == nv(cst.graph)-1 "[Error] ★ is not a valid spanning tree or the graph is not connected!"
    return ★, st_edges; # return a sparse vector
end
function istar(cst::SpanningTree, μ, ★, rng) # corresponds to argmax_{x ∈ X\{★}}⟨μ, x⟩ in the paper
    """ find a max-weight spanning tree differing from ★ """
    st_vec, st_edges = _istar(cst, μ);
    # Case: max-weight spanning tree != ★
    if st_vec != ★
        return st_vec;
    end
    # Case: max-weight spanning tree == ★, then find the 2nd-max-weight spanning tree
    N, K = nv(cst.graph), ne(cst.graph);
    st_g = SimpleGraph(); add_vertices!(st_g, N);
    for e in st_edges
        add_edge!(st_g, src(e), dst(e));
    end
    # randomly assign a node to be the root of st_g
    root = rand(rng, collect(1:N));
    dist_state = dijkstra_shortest_paths(st_g, root);
    # find an edge to swap with st_g
    R = Tuple{Float64, Int64, Int64}[]; # swap_gain, edge_to_add, edge_to_delete
    for (e,k) in cst.emap
        append!(R, _find_swap_edge(cst, μ, e, k, st_vec, dist_state));
    end
    best_idx = argmax([R[i][1] for i=1:length(R)]);
    best_vec = copy(★); best_vec[R[best_idx][2]] = 1; best_vec[R[best_idx][3]] = 0;
    @assert best_vec != ★  "the graph has only 1 spanning tree"
    return best_vec;
end
function _find_swap_edge(cst, μ, e, k, st_vec, dist_state)
    R = Tuple{Float64, Int64, Int64}[]; # swap_gain, edge_to_add, edge_to_delete
    if st_vec[k] == 0 # st_edges + e forms a cycle C -> find an edge in C - e to delete
        lca_edges = _find_edges_to_lca(src(e), dst(e), dist_state); # find lowest_common_ancestor
        best_swap_gain, best_edge_id = typemin(Float64), -1;
        for ce in lca_edges
            delete_edge_id = _lookup_edge(cst.emap, Edge(src(ce), dst(ce)));
            if μ[k] - μ[delete_edge_id] > best_swap_gain
                best_swap_gain = μ[k] - μ[delete_edge_id];
                best_edge_id = delete_edge_id;
            end
        end
        push!(R, (best_swap_gain, k, best_edge_id));
    end
    return R;
end
function _find_edges_to_lca(u, v, dist_state)
    """ find lowest common ancestor """
    edges = []; cur_u, cur_v = u, v;
    while cur_u != cur_v
        if dist_state.dists[cur_u] < dist_state.dists[cur_v]
            push!(edges, Edge(cur_v, dist_state.parents[cur_v]));
            cur_v = dist_state.parents[cur_v];
        elseif dist_state.dists[cur_u] > dist_state.dists[cur_v]
            push!(edges, Edge(cur_u, dist_state.parents[cur_u]));
            cur_u = dist_state.parents[cur_u];
        else
            push!(edges, Edge(cur_u, dist_state.parents[cur_u]));
            push!(edges, Edge(cur_v, dist_state.parents[cur_v]));
            cur_u, cur_v = dist_state.parents[cur_u], dist_state.parents[cur_v];
        end
    end
    return edges;
end
function enumerate_all(cst::SpanningTree) # required by the NaiveMCP oracle
    """ Based on Kirchhoff's Matrix-Tree Theorem """
    B = Graphs.LinAlg.incidence_matrix(cst.graph, oriented=true); # incidence matrix
    N, M = nv(cst.graph), ne(cst.graph);
    cit = combinations(collect(1:M), N-1);
    # check singularity of each (N-1)x(N-1) submatrices of B'
    sp_cnt = 0; # the number of spanning trees
    I, J, V = [], [], Int[];
    for sub in cit
        S = collect(view(B, collect(1:(N-1)), sub));
        if abs(Int(det(S))) == 1 # the (N-1) edges form a spanning tree
            sp_cnt += 1;
            append!(I, fill(sp_cnt, N-1));
            append!(J, sub);
            append!(V, fill(one(Int), N-1));
        end
    end
    decls = sparse(I, J, V, sp_cnt, M);
    return decls; # return an array of sparse vectors
end


""" TO-DO: bipartite matching """