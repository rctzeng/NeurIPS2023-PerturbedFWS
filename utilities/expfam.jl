######################################################################################################
# Source: https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen    #
######################################################################################################
struct Gaussian
    σ2;
end

# convenience
Gaussian() = Gaussian(1);

# KL divergence
rel_entr(x, y) = x==0 ? 0. : x*log(x/y);
dx_rel_entr(x, y) = x==0 ? 0. : log(x/y);
dy_rel_entr(x, y) = -x/y;

d(expfam::Gaussian,    μ, λ) = (μ-λ)^2/(2*expfam.σ2);
dµ_d(expfam::Gaussian,    μ, λ) = (µ-λ)/expfam.σ2
dλ_d(expfam::Gaussian,    μ, λ) = (λ-µ)/expfam.σ2
invh(expfam::Gaussian,    μ, x) = μ + x*expfam.σ2;
sample(rng, expfam::Gaussian,    μ) = μ + sqrt(expfam.σ2)*randn(rng);

# upward and downward confidence intervals
dup(expfam::Gaussian, μ, v) = μ + sqrt(2*expfam.σ2*v);
ddn(expfam::Gaussian, μ, v) = μ - sqrt(2*expfam.σ2*v);