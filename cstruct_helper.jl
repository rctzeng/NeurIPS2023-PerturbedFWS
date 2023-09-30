using Graphs
using SparseArrays # for accessing a submatrix of a sparse matrix

""" Helper functions """
function _get_edge_weights(graph, emap, μ)
    I, J, V = [], [], Float64[];
    for (e,k) in emap
        # i.e., set the weight matrix W[src(e),dst(e)] = W[dst(e),src(e)] = μ[k];
        push!(I, src(e)); push!(J, dst(e)); push!(V, μ[k]);
        push!(I, dst(e)); push!(J, src(e)); push!(V, μ[k]);
    end
    W = sparse(I, J, V, nv(graph), nv(graph));
    return W; # return a sparse matrix
end
function _lookup_edge(emap, e)
    """ lookup undirected edge e in emap """
    if !haskey(emap, e) # in case of emap storing edges in one direction
        return _lookup_diedge(emap, Edge(dst(e),src(e)));
    end
    return emap[e];
end
function _lookup_diedge(emap, e)
    """ lookup direcred edge e in emap """
    if !haskey(emap, e) # in case of emap storing edges in one direction
        return nothing;
    end
    return emap[e];
end
function _compare_edges(e, v1, v2)
    """ check whether the undirected edge e equals to edge (v1, v2) """
    return ((src(e) == v1) && (dst(e) == v2)) || ((src(e) == v2) && (dst(e) == v1));
end
