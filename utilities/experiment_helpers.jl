using Statistics;

function dump_stats(pep, μ, δs, βs, srs, datas, repeats)
    for i in 1:length(δs)
        δ = δs[i];
        β = βs[i];
        data = getindex.(datas, i);
        ⋆ = istar(pep, µ);
        rule = repeat("-", 60);
        println("");
        println(rule);
        println("$pep at δ = $δ");
        println(@sprintf("%27s", "samples"), " ",
                @sprintf("%6s", "err"), " ",
                @sprintf("%5s", "time"), " ",
                join(map(k -> @sprintf("%7s", k), 1:length(μ))),
        );
        println(@sprintf("%-42s", "μ"), join(map(x -> @sprintf("%0.4f   ", x), μ)));
        println(rule);

        for r in eachindex(srs)
            Eτ = sum(x->x[2], data[r,:])/repeats;
            err = sum(x->(dot(μ,x[1]) != dot(μ,⋆)), data[r,:])/repeats;
            tim = sum(x->x[4],data[r,:])/repeats;
            println(@sprintf("%-20s", long(srs[r])),
                    @sprintf("%7.0f", Eτ), " ",
                    @sprintf("%0.4f", err), " ",
                    @sprintf("%3.1f", tim/1e6),
                    join(map(k -> @sprintf("%6.0f", sum(x->x[3][k], data[r,:])/repeats), 1:length(μ)), " ")
            );
            if err > δ
                @warn "too many errors for $(srs[r])";
            end
        end
        println(rule);
    end
end

function _boxes(pep, μ, δ, β, srs, data, repeats)
    xs = permutedims(collect(abbrev.(srs)));
    means = sum(sum.(getindex.(data,2)),dims=2)/repeats;
    boxplot(xs, map(x -> sum(x[2]), data)', label="", notch=true, outliers=false, xtickfontsize=15, ytickfontsize=15); # yaxis=:log) # xguidefontsize=30, yguidefontsize=30, legendfontsize=30
    plot!(xs, means', marker=(:star4,10,:black), label="");
end
