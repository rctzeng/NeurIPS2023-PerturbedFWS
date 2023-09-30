"""
This script modifies the implementation on Graphs.jl
    [prim.jl](https://github.com/JuliaGraphs/Graphs.jl/blob/master/src/spanningtrees/prim.jl)
to support SparseMatrixCSC (instead of its default dense matrix).
"""

using Graphs
using SparseArrays
using DataStructures # for PriorityQueue

function sp_prim_mst(g, distmx)
    nvg = nv(g)
    pq = PriorityQueue()
    finished = zeros(Bool, nvg)
    wt = fill(typemax(Float64), nvg) #Faster access time
    parents = zeros(Int64, nv(g))

    pq[1] = typemin(Float64); wt[1] = typemin(Float64)
    while !isempty(pq)
        v = dequeue!(pq)
        finished[v] = true
        for u in neighbors(g, v)
            finished[u] && continue
            if wt[u] > distmx[u, v]
                wt[u] = distmx[u, v]
                pq[u] = wt[u]
                parents[u] = v
            end
        end
    end
    return [Edge(parents[v], v) for v in vertices(g) if parents[v] != 0]
end
